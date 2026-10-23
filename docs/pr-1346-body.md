## Why

Refactored `example/qvae_mnist` never placed the QVAE or its batches on one device:

- `config.use_cuda` only toggled DataLoader `pin_memory`, not `model.to(...)`;
- `ModelTuner.train` / `test` passed CPU `inputData` through unchanged;
- `MnistQVAE._create_bm()` builds a Boltzmann machine that defaults to **CUDA** when available, while encoder/decoder `nn.Linear` modules and DataLoader tensors stay on **CPU**.

On a CUDA host the first `QVAE.loss` → `_cross_entropy` → `bm(q_prob)` call therefore mixes devices. This is the same **#164** family (nested / default CUDA vs caller tensors), but on the **MNIST trainer example** rather than `torch_plugin` BM kernels.

Related upstream issue: [#306](https://github.com/qboson/kaiwu-pytorch-plugin/issues/306).

## What

- **`Trainer`**: resolve `self.device` from `use_cuda` and `torch.cuda.is_available()`; call `model.to(self.device)` after `create_networks()`; use `.cpu()` before `.numpy()` in reconstruction saves.
- **`ModelTuner`**: move each `inputData` batch to `next(model.parameters()).device` in `train` and `test`.
- **`tests/test_mnist_device_placement.py`** (#164 bar):
  - device-flag resolution;
  - unified submodule placement after `_create_model`;
  - `ModelTuner` batch alignment repro;
  - CUDA `@skipUnless`: documents `create_networks()` alone splitting encoder (CPU) vs BM (CUDA), trainer unifies for `use_cuda` true/false, single train step on GPU;
  - CPU end-to-end smoke (10 epochs, checkpoint).

## Dedupe

| Item | Relationship |
|------|----------------|
| **#1343** | `FeatureExtractor` stale device snapshot; complementary (inference helper vs trainer loop) |
| **#1291** | `qvae_dist_util` Exponential/beta device; library path |
| **#1268** | `torch_plugin` BM tensor placement; different tree |
| **Merged #164** | Pattern reference (`test_bm_device_sync.py`, parent moves, CUDA skipUnless) |

## Validation

```bash
PYTHONPATH=src pytest tests/test_mnist_device_placement.py -v
python3 -m black --check \
  example/qvae_mnist/trainer/trainer.py \
  example/qvae_mnist/trainer/model_tuner.py \
  tests/test_mnist_device_placement.py
PYTHONPATH=src python3 -m pylint tests/test_mnist_device_placement.py
```

## Risk

Low. Example-only placement and batch moves; CPU-only workflows unchanged when `use_cuda=False`. Black reformats the touched trainer modules for style consistency.
