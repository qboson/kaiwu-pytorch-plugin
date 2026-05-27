# `example/qdiffusion`

Examples and workflow scripts for the public `QDiffusion` module.

## Quick Start

```bash
python example/qdiffusion/simple_train_example.py
python example/qdiffusion/simple_generate_example.py
```

Both scripts import `QDiffusion` from `kaiwu.torch_plugin` and show the smallest
useful training and generation paths.

## Files

- `simple_train_example.py`: minimal objective/optimizer loop
- `simple_generate_example.py`: minimal iterative generation loop
- `run_real_full_workflow.py`: full train/eval workflow example
- `rerun_guided_from_checkpoint.py`: guided rerun from saved checkpoints
- `evaluate_esm2_embedding_distance.py`: sequence quality comparison with ESM2
- `data/UP000005640_9606.fasta`: bundled example FASTA
- `graph/*`: diagrams copied from the original case notes

## Notes

- This directory is example-only; the reusable library code lives in
  `src/kaiwu/torch_plugin/qdiffusion.py`.
- Users should import `QDiffusion` from `kaiwu.torch_plugin`, not from local
  helper files.
