<!--
Coordinator: this file is the PR description. GitHub hides this comment.

gh pr edit 1349 --repo qboson/kaiwu-pytorch-plugin --body-file docs/pr-1349-body.md
-->

## Why

`UnsupervisedDBN.device` is a plain attribute captured when the DBN is built. A parent `nn.Module.to` / `_apply` moves each stacked RBM (merged #164 refreshes that RBM's own `device` inside `_apply`) and does **not** call `UnsupervisedDBN.to`. `forward` and `reconstruct` used to build the visible batch with `torch.FloatTensor(data).to(self.device)`, then mix it with `quadratic_coef` / `linear_bias`.

That is the #164 failure mode on the DBN, not on the RBM. Forcing `self.device` back to the old device after the move still crashes: `RuntimeError: Tensor on device meta is not on the expected device cpu!` or `NotImplementedError: Cannot copy out of meta tensor; no data!`.

Open #1334 can coerce the tensor inside `RestrictedBoltzmannMachine.get_hidden`. It does not repair this. `reconstruct_with_rbm` squares `(data_in - visible_recon)` with the **caller** tensor, which was built from the stale DBN attribute.

## What

- `UnsupervisedDBN._parameter_device(module)` — device of the first parameter of that RBM, reread on every call. A parameter-free module falls back to `self.device`. The helper does not write `self.device`.
- `forward` and `reconstruct` place activations on that layer's parameter device. `transform` is `forward`. `reconstruct_with_rbm(..., device=None)` still reads `rbm.device` (#164 keeps that cache aligned); the DBN methods no longer pass the DBN attribute.
- `tests/test_dbn_parameter_device.py`, split the same way as `tests/test_bm_device_sync.py`:
  - **Every machine (CPU / `meta`)**: parameter-free fallback; parent `.to(device)` and `.to(device=)` do not call `UnsupervisedDBN.to` (`direct_to_calls == []`); parent `_apply` to `meta` plus a forced CPU `self.device` still reconstructs, and `get_hidden` sees `meta`; per-layer forward (`cpu` then `meta`); `layer_index=0` on `meta` while the DBN cache is CPU; `layer_index=1` stays on its own parameters while the DBN cache and a sibling layer are `meta`; positional `.to(torch.float64)` and `.to(dtype=torch.float64)` leave the parameter device unchanged, then a forced `meta` cache still places a float32 batch on that device; integer / float16 / float64 / empty / noncontiguous inputs stay finite against a stale `meta` cache. `transform` matches `forward` on the construction device and on the CUDA path below.
  - **CUDA** `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")`: construct-on-CUDA, `parent.to("cpu")` and `parent.to(device="cpu")` do not call `DBN.to`, then a forced CUDA `self.device` still reconstructs and transforms on CPU weights; a dtype-only `parent.to(dtype=torch.float64)` stays on CUDA and does not call `DBN.to`, and a forced CPU cache still places the batch on CUDA.

Float64 **weights** do not run through real `get_hidden`. That method still allocates a float32 buffer, so the matmul raises `expected m1 and m2 to have the same dtype, but got: float != double`. The dtype tests stub `get_hidden`, record the batch device, and let `reconstruct_with_rbm` matmul the float64 hidden state against the float64 weights. The RBM buffer belongs to #1334 / #1268. This PR does not edit `restricted_boltzmann_machine.py`.

**Not in scope:** #242's constructor `device` argument, `hidden_layers_structure` validation, and `UnsupervisedDBN.to` override. #1334 / #1268 RBM and BM tensor creation. #1347 full-BM `forward`. #1348 QVAE latents. #331 `dbn_digits` fine-tune placement.

## Dedupe

| Item | Relationship |
|------|----------------|
| **#242** (open, fixes #241) | Adds `UnsupervisedDBN(device=...)`, validates layer sizes, and overrides `.to()` so a **direct** `dbn.to("cpu")` / `dbn.to(device=...)` writes `self.device` before `nn.Module.to`. That override is not on the parent path: `nn.Module.to` moves a nested DBN through `_apply` and never calls `UnsupervisedDBN.to`. `RecordingDBN.direct_to_calls` stays `[]` after `parent.to`, `parent.to(device=...)`, and `parent._apply`. #242's tests call `.to()` on the DBN itself and expect the attribute to match; they never nest the module, never `_apply` onto `meta`, and never assign `self.device` back to the old device afterwards. Those forced-stale tests still fail if placement reads the attribute, even after a correct `.to()` sync. #242 also treats positional `to(torch.float64)` as a device (`torch.device(dtype)` raises). This branch does not override `.to()`, does not add the constructor argument, and does not copy the structure checks. |
| **#1334** (open) | RBM `get_hidden` / `get_visible` after `_apply`. Even with that coercion, `reconstruct_with_rbm` still subtracts the caller `data_in` built from the DBN device. **No hunk in `restricted_boltzmann_machine.py`.** |
| **#1268** (open) | Parameter-device tensor creation in `abstract_boltzmann_machine.py`, `full_boltzmann_machine.py`, `restricted_boltzmann_machine.py`, and `qvae.py`. **No `dbn.py` hunk.** |
| **#1347** (open) | Full `BoltzmannMachine.forward` / `objective` and `tests/test_bm_forward_parameter_device.py`. Different type. |
| **#1348** (open) | QVAE `energy` / `bm_loss` / `_cross_entropy` and `tests/test_qvae_bm_device.py`. Different module. |
| **#331** (open) | `example/dbn_digits/supervised_dbn_digits.py` fine-tune network. Does not fix `dbn.py` reconstruct. |
| **Merged #164** | Review bar: `tests/test_bm_device_sync.py` (parent move does not call child `.to`, `_apply` / `meta` on every machine, dtype-only move keeps the device, CUDA class behind `@skipUnless`). Each RBM's own `device` is synced there. `UnsupervisedDBN.device` is not. |

## Validation

Forcing `_parameter_device` to return `self.device` produces 10 errors and 2 CUDA skips. The placement tests fail with `RuntimeError: Tensor on device meta is not on the expected device cpu!` or `NotImplementedError: Cannot copy out of meta tensor; no data!`. The construction-device smoke test, the parent-`.to` recording test, and the parameter-free fallback still pass, because they do not build a mismatched batch. With the live parameter device, on a CPU-only machine (Python 3.12.3, torch 2.7.0+cu126, `torch.cuda.is_available()` is false, black 25.1.0, pylint 3.3.7):

| Command | Result |
|---------|--------|
| `PYTHONPATH=src python3 -m unittest tests.test_dbn_parameter_device` | Ran 11 tests, OK (skipped=2) |
| `PYTHONPATH=src python3 -m pytest tests/test_dbn_parameter_device.py tests/test_bm_device_sync.py -q` | 12 passed, 4 skipped, 5 subtests passed |
| `PYTHONPATH=src python3 -m pytest tests/test_rbm.py -q` | 6 passed, 3 subtests passed (file not modified) |
| `python3 -m black --check` on `src/kaiwu/torch_plugin/dbn.py` and `tests/test_dbn_parameter_device.py` | clean (black 25.1.0) |
| `PYTHONPATH=src python3 -m pylint tests/test_dbn_parameter_device.py` | 10.00/10 |
| `PYTHONPATH=src python3 -m pylint src/kaiwu/torch_plugin/dbn.py` | 10.00/10 |
| same `dbn.py` pylint, `--disable=C0114,C0116` (CI flags) | 10.00/10 |

```bash
PYTHONPATH=src python3 -m unittest tests.test_dbn_parameter_device
PYTHONPATH=src python3 -m pytest tests/test_dbn_parameter_device.py tests/test_bm_device_sync.py -q
PYTHONPATH=src python3 -m pytest tests/test_rbm.py -q

python3 -m black --check \
  src/kaiwu/torch_plugin/dbn.py \
  tests/test_dbn_parameter_device.py
PYTHONPATH=src python3 -m pylint tests/test_dbn_parameter_device.py
PYTHONPATH=src python3 -m pylint src/kaiwu/torch_plugin/dbn.py
PYTHONPATH=src python3 -m pylint src/kaiwu/torch_plugin/dbn.py --disable=C0114,C0116
```

CUDA cases are collected and skipped unless `torch.cuda.is_available()`. They were not executed in the CPU-only run above.

## Risk

Low. `forward` and `reconstruct` copy each layer's visible batch onto that RBM's parameter device. No public API change. `self.device` is intentionally left stale; callers that still read the attribute after a parent move will see the construction device.

Do not combine with #242's `dbn.py` edit: #242 rewrites `__init__` and adds `to()`, and this branch inserts `_parameter_device` plus the two call sites. The hunks are adjacent. #1334, #1268, #1347, #1348, and #331 touch other files and do not overlap.

Supersedes tip `1fdf71d8` on `TNHTH:cursor/dbn-activation-device-8d31`.
