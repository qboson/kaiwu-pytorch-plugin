# QDiffusion-NLP working agreement

This directory is a small BM-only example.

- Keep exactly two user-facing workflows: `train_bm.py` and `generate_bm.py`.
- Keep scalar energy, baseline-only generation, paper reproduction, metric
  evaluation, and parameter sweeps in the separate experiments repository.
- Import generic `QDiffusion`, `QDiffusionConfig`, and `EnergyModel` from
  `kaiwu.torch_plugin`; do not copy SDK implementation into this example.
- The frozen text-diffusion checkpoint is an internal proposal dependency, not
  a standalone example workflow.
- Use continuous identity visible values and SA for the supported BM path.
- Preserve compatibility with existing BM checkpoints when changing metadata
  or state-dict layout.
- Validate imports and CLI parsing locally, then run model loading and a small
  CUDA smoke test on the target server.
