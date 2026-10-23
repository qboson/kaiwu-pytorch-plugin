<!--
Coordinator: this file is the PR description. GitHub hides this comment.

gh pr edit 1423 --repo qboson/kaiwu-pytorch-plugin --body-file docs/pr-1423-body.md
-->

## Why

`FIDCalculator` in `example/qvae_mnist/utils/helpers.py` stored `self.device` at construction and sent every preprocessed batch through `processed.to(self.device)`. A parent `nn.Module.to` / `_apply` moves `self.fid` (`FrechetInceptionDistance`) without calling the child `.to` (merged **#164**, `tests/test_bm_device_sync.py`) and without updating that cache. The next `update_batch` then feeds a CPU batch to weights that already left CPU:

`RuntimeError: Tensor on device meta is not on the expected device cpu!`

A meta batch against a stale CPU cache raises:

`NotImplementedError: Cannot copy out of meta tensor; no data!`

Pointing the cache at `cuda` on a CPU-only build does not get as far as the metric. `Tensor.to(torch.device("cuda"))` raises `AssertionError: Torch not compiled with CUDA enabled`.

Restoring cache placement (`FIDCalculator._metric_device` forced back to `self.device` after `parent._apply(meta)`) still raises the meta/CPU `RuntimeError`. The message contains both `meta` and `cpu`. Deleting the override returns to the live-parameter path and does not call child `.to`.

`Module.to(cpu)` cannot copy out of `meta`. Parameters stay on `meta` and child `.to` is not called. The return trip retargets storage with `_apply` and `zeros_like` (the source values are not read). The weights are finite zeros.

Images are cast to `uint8` before the metric. A dtype-only move to `float64` keeps the parameter device, and placement still follows that device, but Inception then raises `Input type (torch.FloatTensor) and weight type (torch.DoubleTensor)`. That dtype error does not mention `meta`. It is a different defect from the construction-cache device mix.

## What

`_metric_device` rereads the live device from `self.fid` parameters (then buffers; construction preference only when the module has neither) on every `update_batch`. `self.device` remains the construction preference. Parent `_apply` and `Module.to` do not go through the metric's `.to`.

`update_batch` preprocesses on the input device, then calls `processed.to(metric_device)`:

- Same device: `Tensor.to` returns the same object, so identity and `data_ptr` are kept, including a noncontiguous view. A meta view is sliced on `meta`. `Tensor.to(meta)` would materialize a contiguous tensor, and every meta tensor reports `data_ptr() == 0`, so meta identity is the Python object.
- A different device is a copy. The caller is not written in place.
- Leaving `meta` for a real device is refused by `Tensor.to` (`Cannot copy out of meta tensor`). The calculator does not invent image values with `zeros_like`. `zeros_like` is only how tests put parameter storage back on CPU.

Poisoning `self.device` to `meta`, or to `cuda` with no GPU, loses while `FrechetInceptionDistance` parameters stay on CPU. A CPU buffer beats a CUDA cache. With no parameters and no buffers, the construction preference is the only remaining device.

`tests/test_fid_calculator_parameter_device.py` matches the #164 split and the #1412 / #1401 / #1400 depth for this one entry point. The CUDA class is behind `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")`.

**Every machine (CPU / `meta`):**

- An unmoved CPU calculator keeps a noncontiguous caller (`is`, `data_ptr`, layout, dtype, values). The real preprocessor output is contiguous `uint8` `(N, 3, 299, 299)`. `Tensor.to` returns that same object. Two identical batches score a finite FID near 0. Child `.to` is not called. `FIDCalculator.device` stays CPU.
- `Tensor.to` on CPU and `meta`, contiguous or not, returns the same object and the same `data_ptr`. Meta `data_ptr` is 0.
- A noncontiguous `uint8` view that already matches the metric (CPU, or meta after `_apply`) is the object `fid.update` receives. `Resize` itself always returns a contiguous tensor, so this view is a preprocessor stand-in. The real resize path is the contiguous identity case above.
- `parent.to("meta")` and `parent.to(device="meta")` do not call child `.to`. Parameters land on `meta`. The cache stays CPU. A noncontiguous CPU caller is unchanged. The seen batch is a different meta tensor.
- `parent._apply(meta)` does the same for a noncontiguous float64 caller. The caller dtype stays float64. The seen image is `uint8` on `meta` (the cast is intentional). Inception weights stay float32. `compute` raises `Tensor.item() cannot be called on meta tensors`. Child `.to` is not called.
- A meta batch that already matches, contiguous or not, keeps object identity. A noncontiguous meta view stays noncontiguous. Meta values are not read.
- `parent.to(dtype=torch.float64)` stays on CPU, does not call child `.to`, and a poisoned `meta` cache does not win. A noncontiguous float64 caller keeps identity. The tensor handed to `fid.update` is the same `uint8` CPU object the preprocessor returned. The real forward then raises the Float/Double dtype error, and that message does not contain `meta`.
- A direct `fid.to(dtype=torch.float64)` is the one recorded `.to` call. It stays on CPU. The same poisoned meta cache does not win.
- Round trip: `_apply(meta)`, then `parent.to(cpu)` raises `NotImplementedError` and leaves parameters on meta, then `_apply` with `zeros_like` returns readable CPU weights (finite, sum 0, float32). The cache is poisoned to `meta` on that CPU stop and still loses. A further `_apply(meta)` is reread. Child `.to` is not called. The noncontiguous caller is unchanged. The CPU stop keeps preprocessor identity and `compute` is finite.
- A move between two `update_batch` calls is reread. The first keeps CPU identity. The second is a distinct meta tensor. The caller is unchanged.
- Poisoning `device` to `meta`, or to `cuda` with no GPU, does not move CPU weights. A noncontiguous caller is unchanged. The seen image keeps CPU identity. The score stays finite and near 0. No `Torch not compiled with CUDA enabled`.
- Forcing `_metric_device` back to `self.device` after `_apply(meta)` reproduces the meta/CPU `RuntimeError`. Deleting the override places the same caller on meta and does not call child `.to`.
- A meta batch against CPU weights, with the cache poisoned to `cuda`, raises `NotImplementedError` on `Tensor.to(cpu)` before `fid.update`. The caller stays on meta. The CUDA cache is not consulted.
- No parameters: a CPU buffer wins over a CUDA cache, and that buffer's device follows `_apply(meta)`. An empty module falls back to the construction preference (`cuda`, then `meta`, then CPU).

**CUDA** `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")` — collected; skipped on this CPU-only VM (same gap as #164):

- `parent.to("cuda")` and `parent.to(device="cuda")` do not call child `.to`. A noncontiguous CPU caller stays put. `update_batch` sees CUDA. The cache stays CPU.
- `parent._apply(cuda)` scores a finite FID. The cache stays CPU.
- A CUDA batch that already matches, contiguous or not, keeps identity. A noncontiguous view stays noncontiguous. `compute` is finite.
- CUDA, then CPU, then CUDA is reread. On the CPU stop the cache is poisoned to `cuda` and still loses. The noncontiguous CPU caller keeps identity. Child `.to` is not called.
- `parent.to(dtype=torch.float64)` after a CUDA `_apply` stays on CUDA and does not call child `.to`. The construction cache stays CPU. A noncontiguous float64 CUDA caller keeps its `data_ptr`. Placement is CUDA `uint8` identity, then the Float/Double dtype error.
- Forcing the cache back to CPU leaves CUDA weights in charge. A noncontiguous CPU caller is unchanged. The score is finite.
- A CUDA batch, contiguous or not, is copied onto CPU weights. Caller storage stays on CUDA. The cache may be `cuda`. The seen batch is CPU.

## Dedupe

| Item | Relationship |
|------|----------------|
| **Merged #164** | Review bar: `tests/test_bm_device_sync.py` (parent move does not call child `.to`, `meta` on every machine, CUDA class behind `@skipUnless`, dtype-only move keeps the device). Same split here. The live location is the metric parameter device, reread every batch. `FIDCalculator.device` stays the construction preference. BM `_apply` sync does not rewrite it. |
| **#1412** (open) | `Trainer` in `example/qvae_cell/trainer.py`. Different file. Same parent-`_apply` / meta / `zeros_like` return / poisoned `cuda` cache. That trainer follows the caller tensor when the module has no parameters. This calculator falls back to a buffer, then the construction preference. |
| **#1401** (open) | `MLPClassifier` in `downstream/classifier.py`. Same example tree, different file. |
| **#1400** (open) | `RBMRunner` in `rbm_digits.py`. Different file. |
| **#1393 / #1378 / #1376…#1343 / #1334 / #1268 / #1291 / #1290** (open) | Other device themes. No hunk in `FIDCalculator.update_batch`. |
| **#567** (open) | Same file: `compute_fid_in_batches` list/tuple acceptance. Does not touch `FIDCalculator.update_batch` / `_metric_device`. This PR does not reformat the pre-existing helper style. |
| **#805 / #1245 / #847 / #565 / #552 / #546 / #342 / #332 / #267** | Same `helpers.py` sampling/plot/CIM isolation. Non-overlapping hunks. |

`FIDCalculator` had no open device PR before this branch.

## Validation

CPU-only VM, CPython 3.12.3, `torch==2.7.0+cpu`. CUDA tests were collected and skipped (`requires CUDA`). 21 tests in this module (14 run, 7 skipped) plus 12 passing subtests. Together with #164: 17 passed, 9 skipped.

`torch-fidelity` (pulled in by `torchmetrics[image]`) ships a top-level `tests` package. While that package is on `sys.path`, `python3 -m unittest tests.test_...` imports it instead of this repo. The module commands below were run with that package moved aside. `python3 tests/test_fid_calculator_parameter_device.py -q` and `python3 -m unittest discover -s tests -p 'test_*device*.py'` pass with the package left installed. Counts match: 14 passed, 7 skipped for this module; 17 passed, 9 skipped with #164.

| Command | Result |
|---------|--------|
| `python3 -m unittest tests.test_fid_calculator_parameter_device -q` | **14 passed, 7 skipped** (CUDA), 12 subtests passed |
| `python3 -m unittest tests.test_fid_calculator_parameter_device tests.test_bm_device_sync -q` | **17 passed, 9 skipped** (this module 14 + 7 CUDA; #164 bar 3 passed + 2 CUDA skipped) |
| `python3 tests/test_fid_calculator_parameter_device.py -q` | **14 passed, 7 skipped** |
| `python3 -m black --check tests/test_fid_calculator_parameter_device.py` | clean (black 26.10.1) |
| `python3 -m black --check example/qvae_mnist/utils/helpers.py` | pre-existing failure, same reformatting diff size as the parent revision. Not reformatted. |
| `python3 -m pylint --rcfile=pylintrc tests/test_fid_calculator_parameter_device.py` | **10.00/10** (pylint 4.1.3) |

```bash
python3 -m unittest tests.test_fid_calculator_parameter_device -q
python3 -m unittest tests.test_fid_calculator_parameter_device tests.test_bm_device_sync -q
python3 -m black --check tests/test_fid_calculator_parameter_device.py
python3 -m pylint --rcfile=pylintrc tests/test_fid_calculator_parameter_device.py
```

## Out of scope

- CUDA class collected; not executed on this CPU-only VM (same gap as #164).
- Copying parameter *values* out of `meta` with `.to()` (that raises; the return trip uses `_apply` and `zeros_like`, so the weights are zeros).
- Casting the batch onto the parameter dtype. The metric receives `uint8`. Float64 weights still raise on the real Inception forward after the device is correct.
- An empty module with no parameters and no buffers. The last resort remains the construction preference, including a poisoned one. A real `FrechetInceptionDistance` always has parameters. #1412 follows the caller instead.
- `compute_fid_in_batches` list inputs (**#567**). `MLPClassifier` (**#1401**). `RBMRunner` (**#1400**). The cell trainer (**#1412**).
- Reformatting pre-existing `helpers.py` black noise.

## Commits (fork)

| SHA | Summary |
|-----|---------|
| `b040aa1750d5181da1ed657c6eb031a19decca85` | place FID updates on the live parameter device |
| branch head | lock the #164 matrix (meta identity, poisoned cuda cache, `zeros_like` return) and sync this PR body |

## Merge-ready checklist

- [x] Fork branch `cursor/fid-calculator-parameter-device-7d2a` pushed to TNHTH
- [x] Harden pass matches the #164 bar (CPU / `meta` executed; CUDA class present and skipped without a GPU)
- [ ] Maintainer review
