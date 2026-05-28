# `example/qdiffusion`

Examples and workflow scripts for the public `QDiffusion` module.

## Data
[data used for example/qdiffusion](https://www.uniprot.org/proteomes/UP000005640)


## Quick Start

```bash
pip install -r example/qdiffusion/requirements.txt
python example/qdiffusion/simple/simple_train_example.py
python example/qdiffusion/simple/simple_generate_example.py
```

These commands work in both common local modes:

- development checkout: run directly from the repo root
- installed package: `pip install -e .` and then run the same commands

## How It Fits Together

This example tree is organized around one simple boundary:

- `src/kaiwu/torch_plugin/qdiffusion.py` contains the generic `QDiffusion` core
- `example/qdiffusion/dplm/` adapts DPLM checkpoints into that generic core
- `example/qdiffusion/simple/` shows the smallest runnable usage

The main assembly path is:

1. a script calls `build_dplm_qdiffusion(...)`
2. `dplm/dplm_builder.py` loads one DPLM proposal backbone and one DPLM feature encoder
3. `dplm/dplm_modeling.py` builds one conditioned RBM reranker on top of that feature encoder
4. the builder constructs one generic `QDiffusion(...)`
5. the script calls `objective(...)` for training or `generate(...)` for inference

## Simple

`simple/` is the smallest runnable surface:

- `simple/simple_train_example.py`: minimal objective/optimizer loop
- `simple/simple_generate_example.py`: minimal iterative generation loop

These scripts still use DPLM under the hood, but only through the example-side
factory helpers in `dplm/`.

Use `simple/` when you want to understand the public API first without reading
the larger experiment workflow.

## DPLM

`dplm/` contains the DPLM-specific adapter layer and larger workflows:

- `dplm/dplm_builder.py`: DPLM-to-`QDiffusion` assembly entrypoint
- `dplm/dplm_modeling.py`: DPLM/ESM modeling, feature encoding, and RBM reranker support code
- `dplm/train_workflow.py`: full train/eval workflow script
- `dplm/rerun_from_checkpoint.py`: guided rerun script from saved checkpoints
- `dplm/eval_esm2_distances.py`: ESM2 distance evaluation script

`dplm/train_workflow.py` is the best place to read the full experiment chain.

Its end-to-end flow is:

1. read and filter FASTA records
2. split them into train/validation/test sets
3. build a `QDiffusion` generator with a DPLM proposal model and an RBM reranker
4. tokenize sequences into `targets`
5. call `generator.objective({"targets": ...})` inside the epoch loop
6. optimize `energy_objective.mean()`, mainly training the RBM reranking path
7. save compact checkpoints containing `energy_encoder`, `feature_projector`, `energy_rbm`, `energy_head`, and `vocab_proj`
8. rebuild baseline and guided generators for test-time generation
9. compare baseline vs guided outputs and write reports

Inside `QDiffusion.objective(...)`, the training path is:

1. start from clean `targets`
2. corrupt them into noisy `x_t`
3. run the proposal model to produce logits
4. sample negative candidates from those logits
5. score positive and negative candidates with the conditioned RBM reranker
6. return `energy_objective` and related tensors to the outer training loop

## Shared Assets

- `data/UP000005640_9606.fasta`: bundled example FASTA
- `graph/*`: diagrams copied from the original case notes

## Notes

- This directory is example-only; the reusable library code lives in
  `src/kaiwu/torch_plugin/qdiffusion.py`.
- Users should import the generic `QDiffusion` core from `kaiwu.torch_plugin`.
- DPLM loading is no longer part of the formal `src` API; the DPLM factory in
  this directory is the example-side compatibility layer.
- The guided path in these examples is now `DPLM proposal + RBM reranker`.
- `simple/` and `dplm/` are designed to be read together:
  `simple/` shows the API surface, while `dplm/` shows the full experiment workflow.
