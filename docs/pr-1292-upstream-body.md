## Why

`QDiffusion` stores `device` (and `dtype`) as plain attributes that are refreshed only when `.to()` runs on the wrapper itself. PyTorch moves nested modules through `_apply` **without** calling a child's `.to()` override—the same mechanism covered for Boltzmann machines in merged #164.

When training or evaluation code wraps a generator in a parent `nn.Module` (DDP, custom containers, checkpoint reload patterns) and calls `parent.to(device)`, parameters relocate but `generator.device` can remain stale. Example scripts and DPLM workflow helpers previously moved token batches with `encoded["input_ids"].to(generator.device)`, leaving inputs on the wrong device while `forward` / `generate` run on parameters.

Library-side metadata sync is tracked in open **#488**. This PR fixes the **example layer** so workflows stay correct before and after #488 lands.

## What

- Add `parameter_device()` in `example/qdiffusion/dplm/utils/runtime.py` and route `encode_sequence()` through it.
- Replace `generator.device` token placement in DPLM workflow collate, ESM2 eval helpers, and the simple train/generate examples.
- Add `tests/test_qdiffusion_example_parameter_device.py` modeled on `tests/test_bm_device_sync.py`:
  - Parent `.to()` / `_apply` mechanism tests (including that child `QDiffusion.to()` is not invoked).
  - Explicit stale-cache repro (`generator.device` vs parameter device).
  - Collate and `build_full_mask_input` coverage.
  - CUDA-only parent `cuda → cpu` encode/collate tests behind `@unittest.skipUnless(torch.cuda.is_available())`.

**Not in scope:** `src/kaiwu/torch_plugin/qdiffusion.py` (owned by #488).

## Dedupe

| Item | Relationship |
|------|----------------|
| **#488** (`qdiffusion.py` `_apply` / metadata) | Complementary: library fix vs example/runtime defense in depth |
| **#1268** BM/RBM/QVAE parameter device | No file overlap |
| **#1275** gbrbm | No overlap |
| **#1290** maifs/plugin | No overlap |
| **#1291** qvae_dist_util | No overlap |
| **#242** dbn | No overlap |
| **Merged #164** | Same *class* of bug (stale plain `device` after parent move); this PR applies the lesson to Q-Diffusion examples |

## Validation

```bash
PYTHONPATH=src python3 -m unittest tests.test_qdiffusion_example_parameter_device -v
```

On a CUDA machine, the `TestExampleDeviceSyncAcrossAccelerators` cases run automatically; on CPU-only CI they are skipped.

```bash
black --check example/qdiffusion/dplm/utils/runtime.py \
  tests/test_qdiffusion_example_parameter_device.py
pylint example/qdiffusion/dplm/utils/runtime.py \
  tests/test_qdiffusion_example_parameter_device.py
```

## Risk

Low: changes are confined to example/runtime helpers and tests; no public API or library behavior changes. Worst case is redundant `.to(parameter_device)` when `generator.device` already matches (no-op).
