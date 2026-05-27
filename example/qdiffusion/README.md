# `example/qdiffusion`

Examples and workflow scripts for the public `QDiffusion` module.

## Quick Start

```bash
pip install -r example/qdiffusion/requirements.txt
python example/qdiffusion/simple/simple_train_example.py
python example/qdiffusion/simple/simple_generate_example.py
```

These commands work in both common local modes:

- development checkout: run directly from the repo root
- installed package: `pip install -e .` and then run the same commands

## Simple

`simple/` is the smallest runnable surface:

- `simple/simple_train_example.py`: minimal objective/optimizer loop
- `simple/simple_generate_example.py`: minimal iterative generation loop

These scripts still use DPLM under the hood, but only through the example-side
factory in `dplm/`.

## DPLM

`dplm/` contains the DPLM-specific adapter layer and larger workflows:

- `dplm/dplm_factory.py`: DPLM-to-`QDiffusion` adapter entrypoint
- `dplm/dplm_runtime.py`: DPLM/ESM runtime support code
- `dplm/run_real_full_workflow.py`: full train/eval workflow example
- `dplm/rerun_guided_from_checkpoint.py`: guided rerun from saved checkpoints
- `dplm/evaluate_esm2_embedding_distance.py`: sequence quality comparison with ESM2

## Shared Assets

- `data/UP000005640_9606.fasta`: bundled example FASTA
- `graph/*`: diagrams copied from the original case notes

## Notes

- This directory is example-only; the reusable library code lives in
  `src/kaiwu/torch_plugin/qdiffusion.py`.
- Users should import the generic `QDiffusion` core from `kaiwu.torch_plugin`.
- DPLM loading is no longer part of the formal `src` API; the DPLM factory in
  this directory is the example-side compatibility layer.
