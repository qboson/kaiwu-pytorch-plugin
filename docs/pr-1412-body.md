<!--
Coordinator: this file is the PR description. GitHub hides this comment.

gh pr edit 1412 --repo qboson/kaiwu-pytorch-plugin --body-file docs/pr-1412-body.md
-->

## Why

`Trainer` in `example/qvae_cell/trainer.py` stored `self.device` at construction and sent every batch through `tensor.to(self.device)`. A parent `nn.Module.to` / `_apply` moves the model parameters without calling the child `.to` (merged **#164**, `tests/test_bm_device_sync.py`) and without updating that cache. The next `run_epoch`, `extract_representation`, or `compute_energy` then feeds a CPU batch to weights that already left CPU:

`RuntimeError: Tensor on device meta is not on the expected device cpu!`

Pointing the cache at `meta` while the weights stay on CPU is the opposite mix:

`RuntimeError: Tensor on device cpu is not on the expected device meta!`

A CUDA cache on a CPU-only build does not get that far. `tensor.to(torch.device("cuda"))` raises `AssertionError: Torch not compiled with CUDA enabled` before the matmul.

Restoring cache placement (`trainer._parameter_device` forced back to `trainer.device` after `parent._apply(meta)`) still raises that `RuntimeError` on all three entry points. The message contains both `meta` and `cpu`. Deleting the override returns to the live-parameter path.

`Module.to(cpu)` cannot copy out of `meta` (`NotImplementedError: Cannot copy out of meta tensor; no data!`). Parameters stay on `meta`. The return trip retargets storage with `_apply` and `zeros_like` (the source values are not read).

The batch dtype is the caller's dtype. A dtype-only move does not change the parameter device, and a matching caller is not recast. This PR does not copy the batch onto the parameter dtype. A float32 batch against float64 weights still dies in the linear op on CPU (`mat1 and mat2 must have the same dtype`). That is a different defect from the construction-cache device mix.

## What

`_parameter_device` reads `next(module.parameters()).device` on every batch. With no parameters it returns the caller tensor's own device, not `self.device`.

`run_epoch`, `extract_representation`, and `compute_energy` place the feature batch and the batch index with `tensor.to(device)`:

- Same device and dtype: `Tensor.to` returns the same object, so identity and `data_ptr` are kept, including a noncontiguous view. A meta view is sliced on `meta`. `Tensor.to(meta)` would materialize a contiguous tensor, and every meta tensor reports `data_ptr() == 0`, so meta identity is the Python object.
- A different device is a copy. The caller is not written in place.
- The batch index stays `torch.long`.
- The feature dtype is not rewritten. A float64 caller stays float64, including against float32 meta weights (meta matmul accepts that mix and the activation dtype follows the input).

`self.device` is still the construction preference. `create_model` still builds and moves the initial `CellQVAE` with it. Assignment does not move an already attached module.

The linear probe is intentional. `CellQVAE.forward` draws an `Exponential` whose beta tensor is **#1291**. These tests stop at batch placement. The probe's `nn.Linear` is a real matmul, so a stale device still raises the runtime error above. Returned numpy values are CPU placeholders from the probe. They are not `CellQVAE` outputs.

`tests/test_cell_trainer_parameter_device.py` matches the #164 split and the #1400 / #1401 depth for three entry points. The CUDA class is behind `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")`.

**Every machine (CPU / `meta`):**

- An unmoved CPU model keeps a noncontiguous caller (`is`, `data_ptr`, layout, dtype, values) through `run_epoch`, `extract_representation`, and `compute_energy`. Activations are finite and on CPU. Representations are `(2, 3)`. Energies have length 2. Metrics are zero. Child `.to` is not called.
- `parent.to("meta")` and `parent.to(device="meta")` do not call child `.to`. Parameters land on `meta`. `trainer.device` stays CPU. A noncontiguous CPU batch is copied. All three entry points see meta. The caller is unchanged.
- `parent._apply(meta)` does the same for a noncontiguous float64 CPU batch. The seen dtype stays float64 (it is not recast onto the float32 weights). The seen tensor is a different object. Child `.to` is not called.
- A meta batch that already matches, contiguous or not, keeps object identity on all three entry points. Meta values are not read.
- float64 weights survive `_apply(meta)`. A noncontiguous float64 CPU batch is copied onto meta as float64. Child `.to` is not called. `trainer.device` stays CPU.
- `parent.to(dtype=torch.float64)` stays on CPU, does not call child `.to`, and a poisoned `trainer.device` of `meta` does not win. A noncontiguous float64 CPU batch keeps identity. Activations are finite float64 on CPU.
- A direct `model.to(dtype=torch.float64)` is the one recorded `.to` call. It stays on CPU. The same poisoned meta cache does not win. The float64 noncontiguous batch keeps identity.
- Round trip: `_apply(meta)`, then `parent.to("cpu")` raises `NotImplementedError` and leaves parameters on meta, then `_apply` with `zeros_like` returns readable CPU weights. The cache is poisoned to `meta` on that CPU stop and still loses. A further `_apply(meta)` is reread. All three entry points follow each leg. Child `.to` is not called. The noncontiguous caller is unchanged. The CPU stop keeps identity.
- A move between two batches is reread by all three entry points. The first batch keeps CPU identity. The second is a distinct meta tensor. Returning the weights to CPU uses `zeros_like` and the weights are finite.
- Poisoning `trainer.device` to `meta`, or to `cuda` with no GPU, does not move CPU weights. All three entry points keep a noncontiguous CPU caller and finite CPU activations.
- With no parameters, a noncontiguous meta batch stays the same object even if `trainer.device` is `meta`, and a noncontiguous CPU batch stays on CPU. A CUDA cache with no parameters does not move a CPU batch (that `.to("cuda")` is what raises `Torch not compiled with CUDA enabled` on the pre-fix path).
- An already-aligned CPU batch still receives `backward` when `trainer.device` is `meta`. Storage is shared. The gradient is finite and nonzero.
- The linear probe names both devices for cache-on-CPU / weights-on-meta and for cache-on-meta / weights-on-CPU.
- Forcing `_parameter_device` back to `trainer.device` reproduces the meta/CPU `RuntimeError` on all three entry points. Deleting the override places a noncontiguous CPU batch on meta and does not call child `.to`.

**CUDA** `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")` — collected; skipped on this CPU-only VM (same gap as #164):

- `parent.to("cuda")` and `parent.to(device="cuda")` do not call child `.to`. A noncontiguous CPU batch stays put. All three entry points see CUDA float32. `trainer.device` stays CPU.
- `parent._apply(cuda)` does the same.
- A CUDA batch that already matches, contiguous or not, keeps identity on all three entry points.
- CUDA, then CPU, then CUDA is reread. On the CPU stop the cache is poisoned to `cuda` and still loses. The noncontiguous CPU caller keeps identity. Child `.to` is not called.
- float64 weights survive a CUDA `_apply`. A noncontiguous float64 CPU batch is copied as float64. It is not recast. Child `.to` is not called.
- `parent.to(dtype=torch.float64)` after a CUDA `_apply` stays on CUDA and does not call child `.to`. The construction cache stays CPU. A noncontiguous float64 CUDA batch keeps identity.
- Forcing the cache back to CPU leaves CUDA weights in charge. A noncontiguous CPU caller is unchanged. All three entry points see CUDA.
- A CUDA batch, contiguous or not, is copied onto CPU weights. Caller storage stays on CUDA. `trainer.device` may be `cuda`. The seen batch is CPU float32 and finite.

## Dedupe

| Item | Relationship |
|------|----------------|
| **Merged #164** | Review bar: `tests/test_bm_device_sync.py` (parent move does not call child `.to`, `meta` on every machine, CUDA class behind `@skipUnless`, dtype-only move keeps the device). The live location here is the parameter device, reread every batch. `Trainer.device` stays the construction preference. BM `_apply` sync does not rewrite it. |
| **#1400** (open) | `RBMRunner.device` in `rbm_digits.py`. Different file. Same parent-`_apply` / meta / CUDA split. That runner also casts NumPy onto the parameter dtype. This trainer does not. |
| **#1401** (open) | `MLPClassifier` predict/proba. Different file. Same split. That estimator casts features onto the parameter dtype because the linear stack requires it. This PR keeps the caller dtype. |
| **#1371 / #1372** (open) | CellQVAE `_encode` / decode in `models.py`. This PR does not edit `models.py`. |
| **#1291** (open) | `qvae_dist_util` Exponential `beta`. Tests use a Linear probe, not real `CellQVAE.forward`. |
| **#1346 / #286 / #485** (open) | MNIST trainer/tuner / checkpoint. Different files. |
| **#1393 / #1378 / #1376 / #1375 / #1374 / #1373 / #1370 / #1349–#1343 / #1334 / #1268 / #1292…** (open) | Other device themes. No hunk in this trainer's batch-placement path. |
| **#174 / #278 / #319 / #402 / #447 / #479 / #502 / #514 / #202** (open) | Same file, non-device edits that still use `.to(self.device)` until rebased. Keep `_parameter_device` if those rebase. `train()` was not reformatted. |

No other open upstream PR edits this Trainer batch-placement path for parameter-device alignment.

## Validation

CPU-only VM, CPython 3.12.3, `torch==2.7.0+cpu`. CUDA tests were collected and skipped (`requires CUDA`). 24 tests in this module (16 run, 8 skipped) plus 10 passing subtests. Restoring `.to(self.device)` at the three call sites fails the module: `FAILED (failures=3, errors=15, skipped=8)`. The two that stay green are the already-aligned CPU path and the probe matmul, which does not call the trainer. The fix was put back before the numbers below.

| Command | Result |
|---------|--------|
| `python3 -m unittest tests.test_cell_trainer_parameter_device -v` | **16 passed, 8 skipped** (CUDA), 10 subtests passed |
| `python3 -m unittest tests.test_cell_trainer_parameter_device tests.test_bm_device_sync -q` | **19 passed, 10 skipped** (this module 16 + 8 CUDA; #164 bar 3 passed + 2 CUDA skipped) |
| `python3 -m black --check tests/test_cell_trainer_parameter_device.py` | clean (black 25.1.0) |
| `python3 -m black --check example/qvae_cell/trainer.py` | pre-existing `train()` wrapping only (**#174 / #278 / #319 / #514**). The new helper is not in that diff. Not reformatted. |
| `python3 -m pylint --rcfile=pylintrc tests/test_cell_trainer_parameter_device.py` | **10.00/10** (pylint 3.3.7) |
| `PYTHONPATH=src python3 -m pylint --rcfile=pylintrc example/qvae_cell/trainer.py` | **7.53/10**. The parent revision of this file, same command and path, was **7.33/10**. Same message codes (`C0114`, `C0301`, `C0411`, `E0401`, `E0611`, `I1101`). The old `compute_energy` line (`C0301`, 108) is gone. One `C0301` remains on the pre-existing early-stop line in `train()`. No new codes. `kaiwu.classical` / `kaiwu.cim` / `kaiwu.preprocess` import-error remains (Kaiwu SDK is not installed here). |

```bash
python3 -m unittest tests.test_cell_trainer_parameter_device -v
python3 -m unittest tests.test_cell_trainer_parameter_device tests.test_bm_device_sync -q
python3 -m black --check tests/test_cell_trainer_parameter_device.py
python3 -m black --check example/qvae_cell/trainer.py
python3 -m pylint --rcfile=pylintrc tests/test_cell_trainer_parameter_device.py
PYTHONPATH=src python3 -m pylint --rcfile=pylintrc example/qvae_cell/trainer.py
```

## Out of scope

- CUDA class collected; not executed on this CPU-only VM (same gap as #164).
- Copying parameter *values* out of `meta` with `.to()` (that raises; the return trip uses `_apply` and `zeros_like`).
- Casting the batch onto the parameter dtype. Caller dtype is preserved. A float32 batch against float64 weights still raises on CPU.
- Real `CellQVAE.forward` / Exponential `beta` (**#1291** / **#1371** / **#1372**).
- Reformatting pre-existing `train()` black noise (**#174 / #278 / #319 / #514**).
- `models.py`, `qvae_dist_util`, `rbm_digits` (**#1400**), the MNIST MLP classifier (**#1401**), the MNIST trainer (**#1346**).

## Commits (fork)

| SHA | Summary |
|-----|---------|
| `e6efc8c4ec4c1ad5ab61e6e5a862911bcbe09fb6` | place trainer batches on the live parameter device |
| branch head | lock the #164 matrix (three entry points, meta identity, poisoned cache) and sync this PR body |

## Merge-ready checklist

- [x] Fork branch `cursor/cell-trainer-parameter-device-d4b7` pushed to TNHTH
- [x] Harden pass matches the #164 bar (CPU / `meta` executed; CUDA class present and skipped without a GPU)
- [ ] Maintainer review
