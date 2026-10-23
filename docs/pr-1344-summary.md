# PR #1344 summary (fork handoff)

**Theme:** `QDEQ.forward` must align caller `x` / `z0` with the wrapped `layer` parameter
device after parent `Module.to()` / `_apply` moves.

**Branch:** `cursor/qdeq-input-device-sync-2912` @ TNHTH/kaiwu-pytorch-plugin  
**Upstream:** qboson#1344 (coordinator-owned; no agent PR to qboson)

**Code:** `src/kaiwu/torch_plugin/qdeq.py` — `_layer_device()`, `x`/`z0` `.to(layer_device)`  
**Tests:** `tests/test_qdeq_device_sync.py` — stale CPU inputs, `parent._apply`→meta,
`RecordingLinearLayer` device assertions, CUDA `skipUnless` pair (#164 parity)

**Detail:** `docs/pr-1344-merge-ready.md`
