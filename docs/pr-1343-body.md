## Summary

- Fix stale-device bug in `example/qvae_mnist/model/feature_extractor.py`: `FeatureExtractor` no longer snapshots `device` at construction; it derives the device from QVAE parameters on each access.
- Add regression tests in `tests/test_feature_extractor_device.py` aligned with #164 / `test_bm_device_sync.py`: stale init-time snapshot repro, direct and parent `Module.to()` moves, CPU/meta mechanism tests, and CUDA `skipUnless` end-to-end paths.

## Problem

After training, workflows often call `qvae.cuda()` / `qvae.cpu()` or move a parent `nn.Module` that owns the QVAE. The old helper cached `self.device = next(qvae_model.parameters()).device` in `__init__`, so `extract()` / `transform()` kept sending batches to the previous device while weights already lived elsewhere (mixed-device `RuntimeError` on CUDA).

## Test plan

```bash
PYTHONPATH=src python3 -m unittest tests.test_feature_extractor_device -v
```

On a CUDA machine, the three `TestFeatureExtractorDeviceCuda` cases should run (not skip).

## Style

```bash
black example/qvae_mnist/model/feature_extractor.py tests/test_feature_extractor_device.py
pylint example/qvae_mnist/model/feature_extractor.py tests/test_feature_extractor_device.py
```
