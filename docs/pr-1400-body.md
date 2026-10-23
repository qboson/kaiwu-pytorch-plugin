<!--
Coordinator: this file is the PR description. GitHub hides this comment.

gh pr edit 1400 --repo qboson/kaiwu-pytorch-plugin --body-file docs/pr-1400-body.md
-->

## Why

`RBMRunner.device` in `example/rbm_digits/rbm_digits.py` was written as a plain attribute at construction. A parent `nn.Module._apply` moves the nested `RestrictedBoltzmannMachine` and refreshes `rbm.device` (merged **#164**, `tests/test_bm_device_sync.py`) without calling the child `.to()`. The runner cache stayed on the construction device. `transform` then placed the batch on that cache while `get_hidden` allocates its state on the live parameter device:

`RuntimeError: Tensor on device meta is not on the expected device cpu!`

Restoring cache placement (`torch.tensor(..., device=runner._device)` into `get_hidden` after `parent._apply(meta)`) still raises that `RuntimeError`. The message contains both `meta` and `cpu`.

`torch.FloatTensor` made two more failures on the same path. On a float32 NumPy array it aliases the caller buffer, so an in-place write changes the caller's data. On a `meta` tensor or a float64 tensor it raises `TypeError`. After `parent.to(dtype=torch.float64)` a float32 batch then dies inside `get_hidden`:

`RuntimeError: expected m1 and m2 to have the same dtype, but got: float != double`

`Module.to(cpu)` cannot copy out of `meta` (`NotImplementedError: Cannot copy out of meta tensor; no data!`). Parameters stay on `meta`. The return trip retargets storage with `_apply` and zeros.

## What

`device` is a property. With parameters it returns `next(rbm.parameters()).device`. Assignment only updates the construction preference `_device` and does not move an RBM that is already attached.

`fit` still does `rbm.to(self._device)` **before** `self.rbm` is replaced, so a second training run uses the construction preference and does not follow weights that were already moved. The old module's `.to` is not called.

`transform` and the `fit` training batch go through `_batch_on_parameter_device`:

- The live parameter device and dtype are read on every call. `_device` is not consulted once an RBM parameter exists.
- `Tensor.to` returns the same object when device and dtype already match, so same-device callers keep identity and `data_ptr`, including a noncontiguous view.
- A different device or dtype is a copy. The caller is not written in place.
- NumPy is copied with `torch.tensor` onto that device and dtype. A float32 array is not aliased.

`plot_images` still uses `images.to(self.device)`. That call now hits the live property. It was not given a dtype cast or any other new behavior.

`get_hidden` is unchanged (**#1334**). It allocates its state with `torch.zeros` (float32), so the NumPy features `transform` returns stay float32 even when the batch and the weights are float64. The batch handed to `get_hidden` matches the parameter dtype, which is what the matmul requires.

`tests/test_rbm_runner_parameter_device.py` matches the #164 split and the #1378 / #1393 pattern. The CUDA class is behind `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")`.

**Every machine (CPU / `meta`):**

- An unmoved CPU runner matches `get_hidden` on a CPU weight copy. The NumPy caller is not mutated.
- A float32 NumPy buffer is not aliased. Writing the tensor `get_hidden` saw does not change the caller. `data_ptr` differs.
- A float32 CPU tensor, including a noncontiguous view, keeps object identity and `data_ptr`.
- Assigning `runner.device` updates `_device` only. Parameter storage and `.to` calls stay put. The property still reports the parameter device.
- With no RBM, `device` follows the setter. Attaching an RBM makes the parameters win.
- `parent.to(meta)` and `parent.to(device=meta)` do not call `rbm.to`. Every parameter lands on `meta`. `rbm.device` and `runner.device` follow. `_device` stays CPU.
- `parent._apply(meta)` does not call `.to`. A CPU NumPy `transform` raises `NotImplementedError` (cannot materialize meta), not the meta/CPU `RuntimeError`. The spy sees meta in and meta out. The caller pointer is unchanged.
- A CPU tensor, including a noncontiguous view, keeps `data_ptr`, dtype, device, and values. `get_hidden` observes a distinct meta tensor.
- A meta batch that already matches the weights keeps storage (`is` and `data_ptr`). Meta values are not read.
- `parent.to(dtype=torch.float64)` stays on CPU, does not call the child `.to`, and `transform` returns finite features. The batch handed to `get_hidden` is float64. The NumPy caller is not aliased.
- A direct `rbm.to(dtype=torch.float64)` is the one recorded `.to` call. It stays on CPU and `transform` returns finite features.
- A float64 tensor that matches float64 weights keeps identity.
- float64 weights survive a later `_apply(meta)` without recasting or rewriting the caller. The batch copied onto `meta` is float64. `get_hidden` still returns a meta tensor. `.to` is not called.
- The property is reread after meta and after the return to CPU. `parent.to(cpu)` raises `NotImplementedError` and leaves every parameter on `meta`. The return trip retargets storage with `_apply` and zeros (the source tensor is not read). A further `_apply(meta)` is reread again. The child `.to` is not called. The CPU score after the zero-storage return is finite.
- `fit` places the new RBM on `_device` even when the previous RBM is on `meta`. The old module is not moved. The training batch is a copy on that fallback device, not an alias of the NumPy buffer.

**CUDA** `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")` — collected; skipped on this CPU-only VM (same gap as #164):

- `parent._apply(cuda)` accepts a CPU NumPy batch, does not call `.to`, and matches a CPU weight copy. The NumPy pointer is unchanged.
- A CPU tensor keeps `data_ptr`. `get_hidden` sees CUDA. Features match the CPU weight copy.
- Forcing the construction preference back to CPU leaves CUDA parameters in charge.
- A batch that is already on CUDA, contiguous or not, keeps storage (`is` and `data_ptr`).
- `parent.to(cuda)` and `parent.to(device="cuda")` do not call the child `.to`. Parameters and `runner.device` land on CUDA. `_device` stays CPU.
- CUDA → CPU → CUDA is reread. `.to` is not called. The CPU stop returns finite features.
- A CUDA batch is copied onto CPU parameters. Caller storage stays on CUDA. Features match the CPU batch.
- float64 weights survive a CUDA `_apply`. The CPU caller keeps `data_ptr` and dtype. The batch `get_hidden` sees is float64 on CUDA. Features are finite.
- `parent.to(dtype=torch.float64)` after a CUDA `_apply` stays on CUDA, does not call the child `.to`, and a float64 CUDA batch keeps identity. Features are finite.

## Dedupe

| Item | Relationship |
|------|----------------|
| **Merged #164** | Review bar: `tests/test_bm_device_sync.py` (parent move does not call child `.to`, `meta` on every machine, CUDA class behind `@skipUnless`, dtype-only move keeps the device). `rbm.device` already syncs. This PR fixes `RBMRunner`'s own construction cache and the batch placed on it. |
| **#1334** (open) | `get_hidden` / `get_visible` in `restricted_boltzmann_machine.py`. This PR does not edit the RBM. The spy still requires the runner's batch to follow live parameters. The float32 state buffer inside `get_hidden` is left as-is. |
| **#1268** (open) | RBM `forward` / Ising / `sample` `_model_device`. Not runner device selection. |
| **#1343** (open) | Same live-property pattern for QVAE `FeatureExtractor`. Different file. |
| **#1393** (open) | FS classifiers (`feature_selection_models.py`). Different file. Same-device storage rule used here. |
| **#1378** (open) | `supervised_dbn_digits` fine-tune. Different file. NumPy-is-copied and dtype-follows-parameters rule used here. No class-index path on `RBMRunner.transform`. |
| **#596** (open) | `plot_images` sample → `get_visible`. `images.to(self.device)` is unchanged in intent and now hits the property. No dtype cast was added. |
| **#565** (open) | CIM checkpoint in `__init__`. The `'./tmp'` quote is the pre-existing black diff. Not reformatted. |
| **#1376 / #1375 / #1370–#1374 / #1344–#1349 / #1290–#1292 / …** (open) | Other files / themes. No `RBMRunner` hunk. |

No other open PR edits `RBMRunner.device` for this cache → parameter-device path.

## Validation

CPU-only VM, CPython 3.12.3, `torch==2.7.0+cpu`. CUDA tests were collected and skipped (`requires CUDA`). 30 tests collected.

| Command | Result |
|---------|--------|
| `python3 -m unittest tests.test_rbm_runner_parameter_device -v` | **20 passed, 10 skipped** (CUDA) |
| `PYTHONPATH=src python3 -m unittest tests.test_rbm tests.test_bm_device_sync -v` | **9 passed, 2 skipped** (CUDA in `test_bm_device_sync`) |
| `python3 -m black --check tests/test_rbm_runner_parameter_device.py` | clean (black 25.1.0) |
| `python3 -m black --check example/rbm_digits/rbm_digits.py` | pre-existing diff only: CIM `'./tmp'` → `"./tmp"` (**#565**). New lines add no hunk. Not reformatted. |
| `python3 -m pylint --rcfile=pylintrc tests/test_rbm_runner_parameter_device.py` | **10.00/10** (pylint 3.3.7) |
| `PYTHONPATH=src python3 -m pylint --rcfile=pylintrc example/rbm_digits/rbm_digits.py` | **6.62/10**. The parent revision of this file, same command and environment, was **6.14/10**. No new message codes. `E1136` on `X_torch` went away. `kaiwu.classical` / `kaiwu.cim` import-error remains (Kaiwu SDK is not installed here). |

```bash
python3 -m unittest tests.test_rbm_runner_parameter_device -v
PYTHONPATH=src python3 -m unittest tests.test_rbm tests.test_bm_device_sync -v
python3 -m black --check tests/test_rbm_runner_parameter_device.py
python3 -m black --check example/rbm_digits/rbm_digits.py
python3 -m pylint --rcfile=pylintrc tests/test_rbm_runner_parameter_device.py
PYTHONPATH=src python3 -m pylint --rcfile=pylintrc example/rbm_digits/rbm_digits.py
```

## Out of scope

- CUDA class collected; not executed on this CPU-only VM (same gap as #164).
- Copying parameter *values* out of `meta` with `.to()` (that raises; the return trip uses `_apply` and zeros).
- `get_hidden` / `get_visible` / `forward` (**#1334** / **#1268**). Returned features stay float32 because `get_hidden` allocates a float32 state.
- `plot_images` beyond the live `device` property. No dtype cast.
- Rewriting the pre-existing black quote in the CIM block (**#565**).
- `restricted_boltzmann_machine.py`, `supervised_dbn_digits`, `feature_selection_models`, `FeatureExtractor`.

## Commits (fork)

| SHA | Summary |
|-----|---------|
| `816fa31efade8ffde97d2809540a861a059914e2` | read the RBM parameter device instead of the init cache |
| `5b224e995d428e57cd2a72e6a16e3aa9f7b6983b` | failing repro: NumPy alias, tensor identity, meta/float64 callers, dtype mix |
| `9fdac3cf922b0a427e8ae08b3b6d688dbbf79f9c` | place batches on the live parameter device and lock the #164 matrix |
| branch head | sync this PR body |

## Merge-ready checklist

- [x] Fork branch `cursor/rbm-runner-parameter-device-5e6a` pushed to TNHTH
- [x] Harden pass matches the #164 bar (CPU / `meta` executed; CUDA class present and skipped without a GPU)
- [ ] Maintainer review
