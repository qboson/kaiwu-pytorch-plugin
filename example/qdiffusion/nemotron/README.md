# Nemotron Contextual-Energy QDiffusion Example

This example composes a frozen Nemotron block-diffusion proposal with a
contextual energy encoder, the public KPP `EnergyModel`/`BoltzmannMachine`, and
the PyPI Kaiwu simulated-annealing optimizer. It includes private-data pair
collection, outcome-pairwise training, versioned checkpoints, guided inference,
and resumable matched Native/BM evaluation.

The repository does not contain model weights or training data, and never
downloads them. All paths, devices, datasets, and checkpoints are supplied
explicitly at runtime.

## Locked method boundary

- Main objective: outcome-pairwise margin loss
- Auxiliary term: `0.01 x NCE`, constraining energy scale only; not strict NCE
- Contextual encoder: 1024 dim, 4 layers, 16 heads
- KPP BM: 512 visible x 256 hidden
- Inference: K=4, energy lambda=2.0, at most 8192 new tokens
- Solver: `kaiwu.classical.SimulatedAnnealingOptimizer`

Legacy in-house BM, LightSA, EDLM-NCE, and Energy-Backbone/LoRA routes are not
part of this example.

```text
private JSONL
    |
    v
prepare_pairs --> same-state OutcomePair (pairs.pt)
                         |
                         v
                    train.py
                         |
             Contextual Encoder + KPP BM + SA
                         |
                         v
                       best.pt
                         |
frozen Nemotron native generate -> ContextualEnergyHook -> ContextualEnergyModel scoring -> token decision
                         |
                         v
                    evaluate.py
              (Native/BM matched + resume)
```

Nemotron's native `generate` remains responsible for cache management,
denoising, and stopping. A plain `QDiffusion` instance is used only to score
the candidates exposed at its transfer-selection point; the example does not
copy or replace the native generation loop.

## Dependencies

Install a CUDA-matched PyTorch first, then from the repository root run:

```bash
pip install -e .
pip install -r example/qdiffusion/nemotron/requirements.txt
```

The solver uses the PyPI `kaiwu` release interface pinned by the repository
dependencies:

```python
from kaiwu.classical import SimulatedAnnealingOptimizer
```

Nemotron checkpoints are loaded with Transformers
`trust_remote_code=True`, so only trusted checkpoints may be used.

## Private data and pair schema

The input is JSONL; each line contains at least:

```json
{"problem_id":"math-train:0","split":"train","problem":"...","answer":"..."}
```

The `split` of training pairs must be `train` or `val`. Eval JSONL does not
require `split`; benchmark test answers must never be used for pair collection
or checkpoint selection. One `problem_id` must not leak across splits.
Tensor pairs use the built-in v2 schema; every record stores the current noisy
block, its token features, the frozen hidden state, and the positive/negative
candidates. Loading and merging check shapes, state hashes, transfer masks,
noise-to-candidate changes, positive/negative candidate differences, and split
leakage. Old v1 pairs cannot be mixed with v2 checkpoints; old checkpoints are
no longer usable either — recollect pairs and retrain.

## Entrypoints

All commands run from the repository root as modules (after the install steps
above). Run the four stages in order: `prepare_pairs` -> `merge_pairs` ->
`train` -> `evaluate`.

| Entrypoint | Purpose |
|---|---|
| `prepare_pairs` | Offline collection: run the frozen model, capture same-state candidate branches, force alternative branches, and label them by final-answer correctness into `(positive, negative)` pairs. Supports sharded runs and resumable collection. |
| `merge_pairs` | Merge independently collected `pairs.pt` artifacts into one, re-validating schema and train/val split consistency. |
| `train` | Train the `ContextualEnergyModel` on merged pairs (outcome-pairwise margin + small NCE regularizer); produces `best.pt` selected on val ranking, resumable via `last.pt`. |
| `evaluate` | Matched Native vs BM-guided decoding on an eval JSONL, scored with boxed-only `math-verify`; append-only resume per strategy. |

## 1. Collect same-state pairs

The collector saves atomically per problem; rerunning with the same
configuration skips already-completed problems.

```bash
python -m example.qdiffusion.nemotron.prepare_pairs \
  --model /path/to/Nemotron-Labs-Diffusion-8B \
  --dataset-jsonl /path/to/private_math.jsonl \
  --split train \
  --output-dir /path/to/pairs_train \
  --device cuda:0 \
  --num-candidates 4 \
  --max-new-tokens 512 \
  --block-length 8

python -m example.qdiffusion.nemotron.prepare_pairs \
  --model /path/to/Nemotron-Labs-Diffusion-8B \
  --dataset-jsonl /path/to/private_math.jsonl \
  --split val \
  --output-dir /path/to/pairs_val \
  --device cuda:0

python -m example.qdiffusion.nemotron.merge_pairs \
  --inputs /path/to/pairs_train/pairs.pt /path/to/pairs_val/pairs.pt \
  --output /path/to/pairs_merged.pt
```

A pair is generated only when, under the same noisy state, the same
block/step, and the same transfer mask, both a correct and a wrong outcome
occur. Benchmark test results must not be used for pair collection or
checkpoint selection.

## 2. Train and resume

The defaults are the locked configuration:

```bash
python -m example.qdiffusion.nemotron.train \
  --pairs /path/to/pairs_merged.pt \
  --output-dir /path/to/contextual_energy_run \
  --device cuda:0 \
  --epochs 10 \
  --batch-size 8
```

Outputs:

- `run_config.json`: environment, data counts, hyperparameters, and source revision
- `last.pt`: model, optimizer, scheduler, history; resumable for further training
- `best.pt`: selected by held-out val pair ranking
- `summary.json`: best epoch and per-epoch metrics

`--resume` is enabled by default. When `last.pt` exists, the run compares the
pair path/size/SHA-256, schema, and model/optimizer parameters, and refuses to
resume on any mismatch while recording the resume environment. `--no-resume`
requires a fresh output directory.

## 3. Native/Qdiffusion inference evaluation

```bash
python -m example.qdiffusion.nemotron.evaluate \
  --model /path/to/Nemotron-Labs-Diffusion-8B \
  --checkpoint /path/to/contextual_energy_run/best.pt \
  --dataset-jsonl /path/to/private_math_eval.jsonl \
  --output-dir /path/to/eval_output \
  --device cuda:0 \
  --strategies native bm \
  --num-candidates 4 \
  --energy-lambda 2.0 \
  --max-new-tokens 8192
```

Each strategy keeps its own JSONL partial and configuration fingerprint.
Interrupted runs continue with the same command; changed checkpoint hashes,
data, decoding parameters, or seeds refuse to mix with old results. The output
records accuracy, tokens, NFE, elapsed time, full text, and candidate
selection statistics.

Final answers are scored with boxed-only, NeMo-compatible `math-verify`.

## Notes

Resume metadata fingerprints private input files by path, size, and SHA-256;
the files themselves are never copied into the repository. See
[README_ZH.md](README_ZH.md) for the same content in Chinese.
