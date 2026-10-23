<!--
Coordinator: this file is the PR description. GitHub hides this comment.

gh pr edit 1373 --repo qboson/kaiwu-pytorch-plugin --body-file docs/pr-1373-body.md
-->

## Why

`DEQForecaster` in `example/qdeq/simple/etth1_common.py` embeds the caller history with `nn.Linear` weights. A parent `nn.Module.to` moves those weights through `_apply` and does not call `DEQForecaster.to` (merged **#164**). The CPU history then mixes with parameters that have already moved inside `embed`:

`RuntimeError: Tensor on device meta is not on the expected device cpu!`

Library **#1344** aligns inputs inside `QDEQ.forward`. This defect is in the example forecaster, **before** tokens enter `QDEQ`. Forcing `_parameter_device` to return the history device still raises that `RuntimeError` on the meta parent-move tests.

## What

- Module-level `_parameter_device(module, fallback)` — device of the first parameter of `module`, reread on every call. No cache. An empty module falls back to `fallback` (`StopIteration` is not raised).
- `DEQForecaster.forward`: `history = history.to(_parameter_device(self.embed, history.device))` before normalize / embed. The caller's tensor is not modified in place. Dtype is unchanged.
- CPU Anderson stays the real solver. Meta tests replace the equilibrium block with an identity stand-in because Anderson calls `Tensor.item()`, which meta tensors reject. The stand-in has no parameters; the failure and the fix are both in the embed / head path.
- `tests/test_deq_forecaster_parameter_device.py`, split the same way as `tests/test_bm_device_sync.py`.

**Every machine (CPU / `meta`):**

- `parent.to(meta)` does not call `DEQForecaster.to`. Embed and head land on `meta`.
- `parent._apply(meta)` accepts a noncontiguous CPU history, does not call `DEQForecaster.to`, and leaves the caller unchanged (`data_ptr`, device, dtype, values, still noncontiguous). The forecast is on `meta` with the caller's dtype.
- float64 weights survive `_apply(meta)` without recasting or rewriting the caller history.
- `parent.to(dtype=torch.float64)` stays on CPU, does not call the child `.to`, and the identity forecast matches a CPU weight replay.
- A direct `.to(dtype=torch.float64)` keeps the real Anderson solver on CPU and finite. The caller history is unchanged.
- The helper is reread after `meta` and after a later `_apply` back to CPU. `.to(cpu)` raises `NotImplementedError: Cannot copy out of meta tensor; no data!` and leaves the module on `meta`, so the return trip retargets storage with `_apply`. A further `_apply(meta)` is reread again. `DEQForecaster.to` is not called.
- Empty modules fall back to the supplied device. Moving only `embed` to `meta` wins over a CPU fallback. The head is left on CPU in that helper check and `forward` is not called, because a split head is not this bug.
- A parameter-free embed and head follow a CPU history and a `meta` history.
- Same-device identity forecasts match a CPU weight replay. Caller storage is unchanged.
- Same-device `.to` does not detach: `forecast.sum().backward()` reaches CPU `embed.weight`, `head.weight`, and a CPU input gradient.

**CUDA** `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")` — collected; skipped on this CPU-only VM (same gap as #164 and the sibling device PRs):

- `parent._apply(cuda)` with the real Anderson solver does not call `DEQForecaster.to`. A CPU history stays put (`data_ptr` and values). The forecast is finite on CUDA.
- The identity forecast matches a CPU weight replay.
- CUDA → CPU → CUDA is reread. `DEQForecaster.to` is not called. Forecasts match a CPU weight replay.
- float64 survives a CUDA move and matches a CPU weight replay.
- `parent.to(dtype=torch.float64)` after a CUDA `_apply` stays on CUDA, does not call the child `.to`, and matches a CPU weight replay.
- Backward from a CPU history reaches CUDA `embed.weight` and `head.weight`, and a CPU input gradient.
- A CUDA history forecasts on a CPU forecaster and matches a CPU weight replay. The caller storage stays on CUDA.

**Not in scope:** library `QDEQ.forward` / `QDEQ._layer_device` (#1344). `CellQVAE.energy` (#500). Anderson `Tensor.item()` on meta (the identity stand-in is the test seam; the solver is not edited). Copying parameters *out of* `meta` with `.to()` (that raises; the return trip uses `_apply`). Other open device PRs' files.

The pandas placeholder used to load the example registers a real `ModuleSpec`. Meta `mean` / `std` import `torch._dynamo`, and `find_spec("pandas")` raises `ValueError: pandas.__spec__ is None` when the stub's spec is missing. That failure is the test harness, not the forecaster. With pandas installed the stub is skipped and the old tests happened to pass.

## Dedupe

| Item | Relationship |
|------|----------------|
| **Merged #164** | Review bar: `tests/test_bm_device_sync.py` (parent move does not call child `.to`, `meta` on every machine, CUDA class behind `@skipUnless`, dtype-only move keeps the device). This PR follows that split for `DEQForecaster.forward`. The live location here is the embed parameter, not `bm.device`. |
| **#1344** (open) | Library `src/kaiwu/torch_plugin/qdeq.py` `QDEQ.forward` / `_layer_device`, plus `tests/test_qdeq_device_sync.py`. This PR does **not** edit those files and does not patch `QDEQ`. The example aligns the history **before** `embed`; tokens then enter `self.deq`. Meta tests replace `deq` with an identity module, so they fail inside `forward` rather than inside `QDEQ`. CPU Anderson uses the real solver only when embed and the equilibrium block already share a device. |
| **#500** (open) | `CellQVAE.energy` dtype/device vs the BM. Different file. |
| **#1372 / #1371** (open) | `example/qvae_cell/models.py`. No overlap. |
| **#1370 / #1348** (open) | Library `qvae.py` reconstruction and BM-facing paths. No overlap. |
| **#1349 / #1347 / #1345 / #1334 / #1275 / #1268** (open) | DBN, full BM `forward`, Ising buffers, RBM state, GBRBM, and library tensor creation. No `etth1_common.py` hunk. |
| **#1346 / #1343 / #1292 / #1291 / #1290 / #488 / #512 / #242** (open) | Other example or library device work. Different files. |

No hunk in `src/kaiwu/torch_plugin/qdeq.py`. Do not rebase onto #1344 to pre-merge it. If both land, keep both: this history move, and #1344's move inside `QDEQ.forward`. When a parent `_apply` moves the whole forecaster they see the same device and the second `.to` is a no-op. They do not rewrite each other.

## Validation

On a CPU-only machine (torch 2.7.0+cpu, Python 3.12, no CUDA, pandas not installed). Forcing `_parameter_device` to return the history device fails the meta forward with `RuntimeError: Tensor on device meta is not on the expected device cpu!`. With the live embed parameter device:

| Command | Result |
|---------|--------|
| `PYTHONPATH=src python3 -m pytest tests/test_deq_forecaster_parameter_device.py -o addopts='' -q` | **11 passed, 7 skipped** (CUDA class) |
| `python3 -m black --check tests/test_deq_forecaster_parameter_device.py` | clean (black 25.1.0) |
| `PYTHONPATH=src python3 -m pylint tests/test_deq_forecaster_parameter_device.py` | **10.00/10** |
| same pylint, `--disable=C0114,C0116` | **10.00/10** |

`black --check example/qdeq/simple/etth1_common.py` still wants wraps that are already present on `main` (line length 88). The `_parameter_device` hunk is not part of that diff, so this branch does not reformat the example. `pylint` on that file reports only the optional `pandas` / `kaiwu.cim` / `kaiwu.classical` imports (8.06/10 here, SDK not installed). Those findings are not from the device change.

```bash
PYTHONPATH=src python3 -m pytest tests/test_deq_forecaster_parameter_device.py -o addopts='' -q
python3 -m black --check tests/test_deq_forecaster_parameter_device.py
PYTHONPATH=src python3 -m pylint tests/test_deq_forecaster_parameter_device.py
PYTHONPATH=src python3 -m pylint tests/test_deq_forecaster_parameter_device.py --disable=C0114,C0116
```

### Test matrix

| Test | Result here |
|------|-------------|
| `test_cpu_anderson_forward_leaves_history_unchanged` | passed |
| `test_parent_to_meta_skips_forecaster_to` | passed |
| `test_parent_apply_accepts_cpu_history` | passed |
| `test_float64_parameters_keep_history_dtype_and_storage` | passed |
| `test_dtype_only_move_keeps_cpu_device` | passed |
| `test_dtype_only_anderson_forward_stays_finite` | passed |
| `test_parameter_device_is_reread_after_meta_round_trip` | passed |
| `test_parameter_device_uses_live_parameters_or_fallback` | passed |
| `test_parameter_free_embed_follows_the_history_device` | passed |
| `test_same_device_identity_matches_weight_replay` | passed |
| `test_same_device_alignment_keeps_autograd` | passed |
| `test_parent_apply_accepts_cpu_history_with_anderson` | skipped (no CUDA) |
| `test_identity_forecast_matches_cpu_weights` | skipped (no CUDA) |
| `test_device_is_reread_across_cuda_and_cpu` | skipped (no CUDA) |
| `test_float64_cuda_move_keeps_dtype` | skipped (no CUDA) |
| `test_dtype_only_move_keeps_cuda_device` | skipped (no CUDA) |
| `test_backward_from_cpu_history_reaches_cuda_weights` | skipped (no CUDA) |
| `test_cuda_history_follows_cpu_parameters` | skipped (no CUDA) |

## Commits (fork)

| SHA | Summary |
|-----|---------|
| `c252b29e83373b43ba11e1d0be87adf7892a6751` | `history.to(embed parameter device)` |
| `ab4b32fcde006846d417e9ffda25a0d737b194c5` | #164-bar tests, including caller `data_ptr` / values, dtype-only, meta round-trip, and a pandas stub spec |

## Merge-ready checklist

- [x] Fork branch pushed to TNHTH with local pytest / black / pylint
- [x] Harden pass to the #164 bar (CPU / `meta` matrix above)
- [ ] CUDA class executed on a GPU (collected and skipped here; 7 tests)
- [ ] Maintainer review

## Risk

Low. One non-in-place `.to` of the history onto the live embed parameter device. No public API change. No edit to `src/kaiwu/torch_plugin/qdeq.py`, so #1344 does not conflict. Head parameters are assumed to share the embed device, which is what a parent `_apply` does to the whole forecaster.
