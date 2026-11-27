# Q-Diffusion for Protein Sequence Generation

> This case study applies the Q-Diffusion framework to protein sequence generation. It demonstrates how DPLM (Protein Language Model) proposal models are combined with a BM energy reranker to improve generated sequence quality. For the overall model hierarchy and how Q-Diffusion fits into KPP's architecture, see [Architecture Overview](architecture_overview.md).

## Overview

By the end of this case study you will be able to:

- Understand how Q-Diffusion combines DPLM proposal models with BM energy reranking
- Train an energy reranker on top of a pre-trained protein language model
- Generate protein sequences and evaluate them with ESM2 embedding distances
- Compare baseline vs guided generation quality

## Running the Example

**Dependencies**:

```{code-block} bash
pip install -r example/qdiffusion/requirements.txt
```

**Run**:

```bash
cd example/qdiffusion
python simple/simple_train_example.py
python simple/simple_generate_example.py
```

Or run the full DPLM workflow:

```bash
python dplm/train_workflow.py
```

**Key files**:

- `simple/simple_train_example.py`: minimal Q-Diffusion training example
- `simple/simple_generate_example.py`: minimal generation example
- `dplm/utils/dplm_builder.py`: DPLM → Q-Diffusion assembly helper
- `dplm/models/bm.py`: BM energy reranker (`BMConditionedEnergyModel`)
- `dplm/workflows/train.py`: full training and evaluation workflow

## How It Works

Q-Diffusion for protein sequences combines a DPLM proposal model with a BM energy reranker:

```
protein sequence  →  DPLM proposal  →  logits  →  sample candidates
                                                          ↓
                                                    BM energy reranker
                                                          ↓
                                              energy-weighted final sequence
```

The key components:

1. **DPLM proposal model**: pre-trained protein language model produces logits for each position.
2. **BM energy reranker**: scores candidate sequences using a Boltzmann Machine energy function.
3. **Energy-guided generation**: candidates are weighted by energy, improving sequence quality.
4. **ESM2 evaluation**: generated sequences are compared to the reference proteome using ESM2 embedding distances.

The training objective optimizes the energy reranker, while the DPLM proposal model remains frozen.

---

## Quick Start

The steps below condense `example/qdiffusion/simple/simple_train_example.py`, the minimal teaching example, with imports omitted; run the script for the complete walkthrough.

**1. Build the generator.** `build_qdiffusion` assembles the frozen DPLM proposal model and the trainable BM energy reranker; both checkpoint arguments are Hugging Face model IDs.

```python
generator = build_qdiffusion(
    proposal_ckpt="airkingbd/dplm_150m",
    energy_ckpt="airkingbd/dplm_150m",
    num_candidates=4,
    freeze_proposal=True,
).to(device)
```

**2. Tokenize one mini-batch.** Encode a few FASTA sequences with the generator's own tokenizer; `targets` is the only tensor the outer loop needs — `objective` creates the noisy states, samples candidates, and builds the energy loss.

```python
targets = generator.tokenizer(
    sequences, return_tensors="pt", add_special_tokens=True, padding=True
)["input_ids"].to(generator.device)
```

**3. Optimize the energy objective.** Only the energy-reranker parameters are trainable; `objective` returns `energy_objective` (plus `logits` and `loss_mask` for monitoring).

```python
optimizer = AdamW([p for p in generator.parameters() if p.requires_grad], lr=1e-4)
for step in range(1, num_steps + 1):
    outputs = generator.objective({"targets": targets})
    loss = outputs["energy_objective"].mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

## Key Code

### 1. Q-Diffusion Model Assembly

The `dplm_builder.py` assembles a Q-Diffusion model from a DPLM proposal model
and a BM energy reranker:

```{literalinclude} ../../../../example/qdiffusion/dplm/utils/dplm_builder.py
:pyobject: build_qdiffusion
```

Key assembly details:
- Proposal model: DPLM protein language model (frozen)
- Energy backend: Boltzmann Machine (trainable)
- Output: the core `kaiwu.torch_plugin.QDiffusion` module described in the next section.

### 2. The QDiffusion Model

`build_qdiffusion` returns the core `kaiwu.torch_plugin.QDiffusion` wrapper. Its constructor wires the four parts:

```{literalinclude} ../../../../src/kaiwu/torch_plugin/qdiffusion.py
:pyobject: QDiffusion.__init__
```

- **proposal_model**: the frozen DPLM backbone that predicts token logits (`freeze_proposal=True` freezes it here).
- **energy_model**: the trainable BM energy reranker that scores candidates.
- **token_spec**: special-token metadata (`mask_id`, `pad_id`, `bos_id`, `eos_id`).
- **config**: a `QDiffusionConfig` with generation and training defaults.

The wrapper exposes both training and decoding APIs:

| API | Purpose |
|---|---|
| `objective(batch)` | Builds the one-step training objective; returns logits, masks, and `energy_objective`. |
| `generate(input_tokens, max_steps=500, …)` | Runs the full iterative decoding loop. |
| `step(state, partial_masks=None)` | Runs one denoising/reranking step. |
| `proposal(noisy_tokens)` / `energy(noisy, candidate)` | Direct calls to the two backbones. |
| `tokenizer` | The tokenizer carried by `token_spec`. |

**Key config parameters** (`QDiffusionConfig`):

| Parameter | Default | Meaning |
|---|---|---|
| `num_diffusion_timesteps` | 500 | Number of discrete noising steps in the objective. |
| `num_candidates` | 1 | Proposal candidates sampled per decode step. |
| `proposal_temperature` | 0.0 | Temperature for proposal-side sampling. |
| `proposal_noise_scale` | 1.0 | Gumbel noise scale during proposal sampling. |
| `energy_temperature` | 1.0 | Temperature converting energies into rerank weights. |
| `disable_resample` | False | Disables repetition-collapse resampling. |
| `resample_ratio` | 0.25 | Frequency threshold that triggers resampling. |
| `resample_top_p` | 0.95 | Top-p cutoff used during resampling. |
| `decoding_strategy` | "reparam-uncond-deterministic-linear" | Skeptical-remasking strategy string. |

The Quick Start passes `num_candidates=4`, overriding the default of 1. The remaining defaults are kept.

### 3. The EnergyModel Interface

The energy side is the `kaiwu.torch_plugin.EnergyModel` interface. Its scoring chain turns token candidates into BM-space energies:

```{literalinclude} ../../../../src/kaiwu/torch_plugin/qdiffusion.py
:pyobject: EnergyModel.score_visible_logits
```

1. **Discretize**: `discretize_visible_state` maps visible logits to BM visible conditions via sigmoid.
2. **Sample**: `sample_hidden_state` draws BM hidden states with `condition_sample` for each visible assignment.
3. **Score**: the BM energy of each full state is averaged per candidate.
4. **Diagnose**: `get_last_stats` returns sampling diagnostics (visible/hidden on-ratios).

`score_conditioned` is the abstract entry point; subclasses implement it. This example uses `BMConditionedEnergyModel`, which bridges token space to BM space and then calls the chain above:

```{literalinclude} ../../../../example/qdiffusion/dplm/models/bm.py
:pyobject: BMConditionedEnergyModel.score_conditioned
```

The scoring chain is backend-agnostic: the hidden-state sampler is supplied at construction time, so the same interface supports classical and quantum samplers.

### 4. Training Objective

The Q-Diffusion objective combines proposal logits with energy reranking:

```{literalinclude} ../../../../example/qdiffusion/simple/simple_train_example.py
:pyobject: main
```

Key training details:
- Proposal model: frozen, produces logits
- Energy reranker: trainable, scores candidates
- Loss: energy objective mean, backpropagated through the reranker

### 5. Generation and Evaluation

The generation loop samples from the proposal model and reranks with energy:

```{literalinclude} ../../../../example/qdiffusion/simple/simple_generate_example.py
:pyobject: main
```

## Training and Evaluation

- **Training curves**: energy objective loss over epochs.
- **Generated samples**: protein sequences generated by the guided workflow.
- **ESM2 distance**: cosine and L2 distances between generated and reference proteome embeddings.
- **Baseline vs guided**: compare baseline (DPLM only) vs guided (DPLM + BM reranker) quality.

---

## Summary

- Q-Diffusion combines DPLM proposal models with BM energy reranking for protein generation.
- The energy reranker is trained on top of a frozen protein language model.
- Guided generation improves sequence quality measured by ESM2 embedding distances.
- The Q-Diffusion pipeline combines a frozen DPLM proposal model with a trainable BM energy reranker, and the same assembly pattern transfers to other proposal models.
