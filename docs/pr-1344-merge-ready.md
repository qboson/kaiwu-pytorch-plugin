# PR #1344 — QDEQ input device alignment (merge-ready notes)

Upstream theme: [qboson/kaiwu-pytorch-plugin#1344](https://github.com/qboson/kaiwu-pytorch-plugin/pull/1344)  
Fork branch: `cursor/qdeq-input-device-sync-2912` on `TNHTH/kaiwu-pytorch-plugin`  
Coordinator gate: no direct qboson PR edits from this agent; fork-only pushes.

## Problem

After `parent.qdeq = QDEQ(layer, …)` and `parent.to(device)` (or the equivalent
`parent._apply` path PyTorch uses internally), layer parameters may already sit on
`meta` or CUDA while callers still pass CPU `x` / `z0`. Before the fix,
`QDEQ.forward` forwarded those tensors into `layer(z, x)` unchanged, yielding
CPU outputs on `meta` or raising mixed-device `RuntimeError` on CUDA.

## Fix

- Add `QDEQ._layer_device(x)` (first layer parameter device, else `x.device`).
- In `forward`, `x = x.to(layer_device)` and align optional `z0` before solve
  and the final differentiable layer call.

## Test bar (#164 alignment)

| Case | File | Host |
|------|------|------|
| Stale CPU `x` after `parent.to(meta)` | `tests/test_qdeq_device_sync.py` | all |
| Stale CPU `x` after `parent._apply(→meta)` | same | all |
| Stale CPU `z0` with `deq._apply(→meta)` | same | all |
| No-parameter layer falls back to `x.device` | same | all |
| CUDA params + CPU `x` | same | CUDA only (`skipUnless`) |
| Construct on CUDA → `parent.to(cpu)` → forward | same | CUDA only |

Recording layers assert every `layer(z, x)` call sees activations on the
parameter device, not only the returned tensor.

## Verification commands

```bash
PYTHONPATH=src python3 -m black src/kaiwu/torch_plugin/qdeq.py tests/test_qdeq_device_sync.py
PYTHONPATH=src python3 -m pylint src/kaiwu/torch_plugin/qdeq.py tests/test_qdeq_device_sync.py
PYTHONPATH=src python3 -m pytest tests/test_qdeq_device_sync.py -v
```

## Commits (fork)

| SHA | Summary |
|-----|---------|
| `b1c8cb2e212d0cb5f436ac6a2a3828b095bd07d6` | Initial `QDEQ._layer_device` + input alignment |
| `e4f75a340395958092390fd8ce25b13a5485444a` | Strengthened #164-style regression matrix + docs |

## Merge-ready checklist

- [ ] Fork branch pushed to TNHTH with green local pytest/black/pylint
- [ ] Upstream serial gate opens for qboson cherry-pick / PR update
- [ ] Full `tests/test_qdeq.py` suite in CI (requires `kaiwu` SDK with
      `conversion`); not expanded here per T-0020 scope
