# Minimal Q-Diffusion with a Custom Energy Model

`QDiffusion` generates discrete sequences by sampling proposal candidates and
**reranking them with an energy model**. The energy side is a public extension
point (`kaiwu.torch_plugin.EnergyModel`), but the bundled example depends on a
DPLM checkpoint plus Hugging Face downloads. This example keeps only the mechanism:

| Component | What it does here |
| --- | --- |
| `ToyProposal` | an embedding table that maps each token to a fixed logits row |
| `BagOfTokensEnergy` | an `EnergyModel` subclass that scores a candidate by its token weights minus a penalty for dropping the observed context |
| `SequenceTokenSpec` | padding/BOS/EOS/mask ids for a six-token vocabulary |
| `QDiffusion.generate` | runs the iterative decode loop over a partially masked sequence |

The script reports how often the energy model was consulted and whether every
mask position was decoded, so the reranking step is visible instead of implicit.

## Run

```bash
python example/qdiffusion_minimal/qdiffusion_minimal.py
```

Expected output (seed 0):

```text
steps=4 seed=0 context_weight=1.5
generated tokens        : [1, 4, 4, 4, 4, 2]
energy-model calls      : 4
ordinary-token fraction : 1.00
PASS: the energy model reranked every candidate and the mask was fully decoded
```

`--steps` controls the decode iterations, `--seed` makes the run reproducible and
`--context-weight` changes how strongly the energy model rewards keeping the
observed tokens.

## Writing your own energy model

Subclass `EnergyModel` and implement `score_conditioned(noisy_tokens,
candidate_tokens, attention_mask)`; return one energy per candidate with shape
`[batch, 1]`. Lower energies are preferred by the reranking step. That is exactly
what `BagOfTokensEnergy` does in `qdiffusion_minimal.py`.

## Dependencies

Only the project requirements (`torch`); no checkpoint, no Hugging Face download,
no Kaiwu SDK credentials and no GPU are needed.
