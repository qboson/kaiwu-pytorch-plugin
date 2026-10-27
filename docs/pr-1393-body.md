<!--
Coordinator: this file is the PR description. GitHub hides this comment.

gh pr edit 1393 --repo qboson/kaiwu-pytorch-plugin --body-file docs/pr-1393-body.md
-->

## Why

`TinyCNN`, `SimpleRNN`, and `SimpleLSTM` in `example/feature_selection/feature_selection_models.py` fed caller batches straight into conv, recurrent, and linear layers. A parent `nn.Module._apply` moves those weights without calling the child `.to()` (merged **#164**, `tests/test_bm_device_sync.py`). A CPU batch then mismatches `meta` or CUDA parameters:

- `TinyCNN`: `RuntimeError: Tensor on device meta is not on the expected device cpu!`
- `SimpleRNN` / `SimpleLSTM`: `RuntimeError: Input and parameter tensors are not at the same device, found input tensor at cpu and parameter tensor at meta`

Restoring the caller-device input (`_on_parameter_device` returns the batch unchanged) reproduces that `RuntimeError` on `parent._apply(meta)` for all three classifiers. The message contains both `meta` and `cpu`.

## What

Each classifier `forward` reads the live parameter device on every call and copies the input to that device only. Dtypes stay put: a float64 weight move does not recast a tensor the caller still owns, and a same-dtype batch is what the conv / RNN / LSTM op requires. Caller storage is not rewritten. `Tensor.to` returns the same object when the device already matches, so same-device callers keep identity and `data_ptr`, including a noncontiguous view.

Conv and recurrent activations stay on that device, so the linear head does not need a second move. These modules have no embedding index, attention mask, or dropout input (`dropout=0`, one recurrent layer). The head is the second parameter group, and the tests spy both stages.

`tests/test_feature_selection_parameter_device.py` matches the #164 split and the #1376 / #1378 pattern. The CUDA class is behind `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")`.

**Every machine (CPU / `meta`):**

- An unmoved CPU model reuses the caller tensor (`is` and `data_ptr`). Logits are finite. The first stage and the linear head both see that device.
- `parent.to(meta)` and `parent.to(device=meta)` do not call the child `.to`. Every parameter lands on `meta`. There are no buffers.
- `parent._apply(meta)` does not call `.to`. A CPU batch keeps `data_ptr`, dtype, device, layout, and values. The first stage observes a distinct meta tensor. The head observes meta activations. Logits are meta and keep the caller dtype.
- A noncontiguous CPU view keeps storage and stays noncontiguous, on CPU and again after `parent._apply(meta)`.
- A meta batch that already matches the weights keeps storage (`is` and `data_ptr`). Meta values are not read.
- `parent.to(dtype=torch.float64)` stays on CPU, does not call the child `.to`, and scores finite float64 logits. The matching caller batch keeps identity.
- A direct `.to(dtype=torch.float64)` is the one recorded `.to` call. It stays on CPU and scores finite float64 logits.
- float64 weights survive a later `_apply(meta)` without recasting or rewriting the caller batch. Logits are float64 on `meta`.
- The helper is reread after meta and after the return to CPU. `Tensor.to(cpu)` and `Module.to(cpu)` raise `NotImplementedError` and leave every parameter on `meta`. The return trip retargets storage with `_apply` and zeros (the source tensor is not read). A further `_apply(meta)` is reread again. The child `.to` is not called. The CPU score after the zero-storage return is finite.
- An empty module follows a CPU tensor and a meta tensor (`is`). Real weights win over the opposite fallback, for all three classifiers.
- Same-device alignment keeps autograd. `sum().backward()` reaches the CPU input and every parameter, including recurrent weights and the linear head. Gradients are finite and nonzero.
- No `Embedding` or `Dropout` submodule is present.

**CUDA** `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")` — collected; skipped on this CPU-only VM (same gap as #164):

- `parent._apply(cuda)` and `parent.to("cuda")` do not call `.to`. CPU batches score on CUDA, keep caller storage, and match the pre-move CPU logits. Both stages see the parameter device.
- A batch that is already on CUDA, contiguous or not, keeps storage (`is` and `data_ptr`). Logits are finite on that device.
- CUDA → CPU → CUDA is reread. `.to` is not called. Each stop matches the CPU reference.
- float64 survives a CUDA `_apply`. Caller storage stays put. Logits stay float64 and match the CPU float64 reference.
- `parent.to(dtype=torch.float64)` after a CUDA `_apply` stays on CUDA, does not call the child `.to`, and keeps float64 logits.
- Backward from a CPU batch reaches every CUDA parameter, including the head. The input gradient stays on CPU and is finite and nonzero. Caller `data_ptr` stays put.
- A CUDA batch is copied onto CPU parameters. Caller storage stays on CUDA. Logits match the CPU batch.

## Dedupe

| Item | Relationship |
|------|----------------|
| **Merged #164** | Review bar: `tests/test_bm_device_sync.py` (parent move does not call child `.to`, `meta` on every machine, CUDA class behind `@skipUnless`, dtype-only move keeps the device). This PR follows that split for the three feature-selection classifiers. The live location is the parameter device, reread every call. |
| **#1378** (open) | `supervised_dbn_digits.py` fine-tune, including class-index dtype. Different file. This PR does not cast caller dtype. |
| **#1376 / #1375** (open) | `esm_patch.py` / `bm.py`. Different files. |
| **#1374 / #1373 / #1372 / #1371 / #1370** (open) | Other example modules. Different files. |
| **#1349–#1343 / #1334 / #1292…** (open) | Library / other examples. Unchanged. |
| **#1290** (open) | MAIFS wrapper places the training batch on the parameter device. This PR fixes the three classifiers' own `forward`. |
| **#1179** (open) | Same file, dimension checks in `__init__` only. This PR touches `forward` docstrings and the end-of-file device helpers. |
| **#259** (open) | `common.py` `masked_mean_pool`. Different file. |

No other open PR edits these three classifier `forward` paths for parameter-device alignment.

## Validation

CPU-only VM, CPython 3.12.3, `torch==2.7.0+cpu`. CUDA tests were collected and skipped (`requires CUDA`). 19 tests collected.

| Command | Result |
|---------|--------|
| `python3 -m unittest tests.test_feature_selection_parameter_device -v` | **12 passed, 7 skipped** (CUDA). CPU cases subtest `TinyCNN`, `SimpleRNN`, and `SimpleLSTM`. |
| `python3 -m black --check` on both files | clean (black 25.1.0) |
| `python3 -m pylint --rcfile=pylintrc` on both files | **10.00/10** (pylint 3.3.7) |

```bash
python3 -m unittest tests.test_feature_selection_parameter_device -v
python3 -m black --check \
  example/feature_selection/feature_selection_models.py \
  tests/test_feature_selection_parameter_device.py
python3 -m pylint --rcfile=pylintrc \
  example/feature_selection/feature_selection_models.py \
  tests/test_feature_selection_parameter_device.py
```

## Out of scope

- CUDA class collected; not executed on this CPU-only VM (same gap as #164).
- Copying parameter *values* out of `meta` with `.to()` (that raises; the return trip uses `_apply` and zeros).
- Casting the caller dtype to the parameter dtype. `parent.to(dtype=torch.float64)` keeps the parameter device and the caller's dtype. A float32 batch against float64 weights still fails inside conv / RNN / LSTM. #1378 casts because its inputs are NumPy.
- MAIFS wrapper (#1290), dim checks (#1179 `__init__`), other open device PRs.
- Embedding, index, mask, and dropout paths: they are not in these three modules. The tests fail if an `Embedding` or `Dropout` submodule appears.

## Commits (fork)

| SHA | Summary |
|-----|---------|
| `7277a4a28719308b54999991d2e092b7e740e703` | place classifier inputs on the live parameter device |
| branch head | lock the #164 / #1376 / #1378 bar for all three classifiers |

## Merge-ready checklist

- [x] Fork branch `cursor/fs-classifier-parameter-device-4c8a` pushed to TNHTH
- [x] Harden pass matches the #164 bar (CPU / `meta` executed; CUDA class present and skipped without a GPU)
- [ ] Maintainer review
