<!--
Coordinator: this file is the PR description. GitHub hides this comment.

gh pr edit 1424 --repo qboson/kaiwu-pytorch-plugin --body-file docs/pr-1424-body.md
-->

## Why

`AbstractSupervisedDBNClassifier.get_layer_activations` in `example/dbn_digits/supervised_dbn_digits.py` placed every visible batch with:

`torch.FloatTensor(X_data).to(self.unsupervised_dbn.device)`

`unsupervised_dbn.device` is a plain construction-time attribute. A parent `nn.Module.to` / `_apply` moves the nested RBM parameters (merged **#164** refreshes each RBM's own `device` inside `_apply`) without calling the child `.to()` and without refreshing the DBN cache. The next `get_hidden` allocates its state on the live parameter device and assigns a CPU batch into it:

`RuntimeError: Tensor on device meta is not on the expected device cpu!`

Poisoning the cache to `meta` while the parameters stay on CPU is the opposite mix and raises `NotImplementedError: Cannot copy out of meta tensor; no data!`. A CUDA cache on a CPU-only build raises `AssertionError: Torch not compiled with CUDA enabled` before the matmul. A dtype-only `parent.to(dtype=torch.float64)` keeps the device but leaves a float32 batch against float64 weights (`float != double`).

The caller path had a second gap. `X.astype(np.float32)` sent every tensor through NumPy: a tensor already on the parameter device lost its identity and `data_ptr`, and `torch.FloatTensor` on a float32 NumPy buffer aliased the caller. Restoring cache placement reproduces the meta/CPU `RuntimeError`; the live parameter device is the only correct source.

## What

`_batch_on_rbm_parameters` rereads `next(rbm.parameters()).device` and `.dtype` on every call. Only a parameter-free RBM falls back to the construction cache (a non-`torch.device` value is ignored). The helper does not write the cache.

- A tensor is placed with `Tensor.to(device=device, dtype=dtype)`. A tensor that already matches is returned as the same object, keeping identity, `data_ptr`, and contiguity, including a noncontiguous view. A device or dtype change is a copy. The caller is never written in place.
- NumPy and list inputs go through `torch.tensor`, which copies; the caller buffer is never aliased.
- `get_layer_activations` passes a tensor through unchanged. Every other input is normalized with `np.asarray(..., dtype=np.float32)`.
- The loop already rereads each layer's own RBM, so a move of the second layer does not change the first layer's placement. `layer_index`, early return, and `return_all_layers` are unchanged.

This PR does **not** change the fine-tune batch path owned by **#1378** (the `_fine_tuning` / `_predict_with_fine_tuning` / `_predict_proba_fine_tuning` lines that still read `unsupervised_dbn.device`). It does not edit `src/kaiwu/torch_plugin/dbn.py` (**#1349** / **#242**) or `restricted_boltzmann_machine.py` (**#1334** / **#1268**).

`tests/test_dbn_layer_activation_parameter_device.py` matches the #164 split. The CUDA class is behind `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")`.

**Every machine (CPU / `meta`), 17 tests plus 12 sub-cases:**

- An unmoved CPU stack matches `get_hidden`. A poisoned `meta` cache stays poisoned and does not change placement; the batch `get_hidden` sees is CPU. The NumPy caller keeps values and address.
- Poisoning the cache to `meta`, or to `cuda` with no GPU, loses to CPU parameters. No `Torch not compiled with CUDA enabled`.
- `parent._apply(meta)` does not call `rbm.to`; every RBM parameter lands on `meta`; `rbm.device` follows through #164; the DBN cache stays CPU. A NumPy activation call raises `NotImplementedError` (cannot materialize meta), not the meta/CPU `RuntimeError`; the spy sees `meta`; the caller is unchanged.
- `parent.to(meta)`, `parent.to(device=meta)`, and `parent.to("meta")` do not call the child `.to`.
- `parent.to(cpu)` after `meta` raises `NotImplementedError` and leaves the parameters on `meta`; the child `.to` is still not called.
- `parent.to(dtype=torch.float64)` stays on CPU, does not call the child `.to`, and the batch follows the parameter dtype. A poisoned `meta` cache does not win. Results match a float64 weight copy.
- Round trip: `_apply(meta)`, rejected `parent.to(cpu)`, `_apply` retarget with `zeros_like` (source values are not read; weights are finite zeros), then `_apply(meta)` again. Each leg is reread; the cache is poisoned to `meta` on the CPU stop and still loses. The caller is unchanged and the child `.to` is never called.
- Restoring cache placement (`_batch_on_rbm_parameters` forced back to `unsupervised_dbn.device`) reproduces the meta/CPU `RuntimeError`. Deleting the override returns to the live-parameter path.
- A float32 NumPy buffer is not aliased: the tensor handed to `get_hidden` has a different `data_ptr`, and mutating it does not change the caller. A nested list takes the same `torch.tensor` path and matches the array result.
- A same-device float32 CPU tensor, contiguous or a noncontiguous view, keeps identity, `data_ptr`, layout, and values. A meta tensor already matching meta weights keeps identity (`data_ptr() == 0`).
- A float64 caller against float32 weights is presented as a float32 copy; the caller stays float64.
- A meta batch against CPU weights with a CUDA cache raises `NotImplementedError` on the way out of meta before `get_hidden`; the caller stays on meta and the cache is not consulted.
- Multi-layer `[3, 2]`: both layers run on CPU, `return_all_layers` keys are `{0, 1}` with shapes `(2, 3)` and `(2, 2)`, `layer_index=0/1` matches, `layer_index=0` stops before the second layer, and an out-of-range index raises `ValueError`. Moving only the second RBM to `meta` leaves the first layer on CPU and makes only the second layer fail.
- No parameters: a `torch.device` cache is the fallback; `None`, `"meta"`, and `"cuda"` strings fall back to CPU. A real RBM always has parameters, so this path is unreachable in practice.

**CUDA** `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")` — collected; 6 tests skipped on this CPU-only VM (same gap as #164):

- `parent._apply(cuda)` and `parent.to("cuda")` / `parent.to(device=cuda)` do not call the child `.to`; CPU batches score on CUDA and match a CPU weight copy; the DBN cache stays CPU.
- A CUDA batch that already matches, contiguous or not, keeps identity and `data_ptr`.
- A CPU cache does not pull CUDA weights back to the host.
- `parent.to(dtype=torch.float64)` after a CUDA move stays on CUDA and follows the float64 dtype.
- CUDA, then CPU, is reread; the CPU stop ignores a poisoned CUDA cache and matches the original CPU weight copy.

## Dedupe

| Item | Relationship |
|------|----------------|
| **Merged #164** | Review bar: `tests/test_bm_device_sync.py` (parent move does not call child `.to`, `meta` on every machine, CUDA class behind `@skipUnless`, dtype-only move keeps the device). Same split here. `rbm.device` already syncs through `_apply`; this PR fixes the DBN construction cache and the batch placed on it. |
| **#1378** (open) | Same file: fine-tune features and class indices. Its body explicitly leaves `get_layer_activations` out of scope (**#1367**). This PR does not touch `_fine_tuning`, `_predict_with_fine_tuning`, `_predict_proba_fine_tuning`, `_align_fine_tune_*`, or `_train_fine_tune_network`. |
| **#1367** (open) | Test-only PR pinning CPU layer-activation extraction (keys, early return, `ValueError`) with a stub RBM. No device or dtype handling, no source hunk, different test file (`tests/test_dbn_layer_activations.py`). The two suites are complementary and do not conflict. |
| **#1349** (open) | `UnsupervisedDBN.forward` / `reconstruct` / `transform` in `src/kaiwu/torch_plugin/dbn.py`. Different file: the outer DBN's own batch placement. This PR does not edit `dbn.py`. |
| **#242** (open) | `UnsupervisedDBN` constructor `device` argument and `.to()` sync in `dbn.py`. Different file and path. This PR does not add or change `.to()`. |
| **#1334** (open) | RBM `get_hidden` / `get_visible` coercion. `restricted_boltzmann_machine.py` is unchanged. This PR only supplies a correctly placed batch. |
| **#1268** (open) | RBM `forward` / Ising / sampling tensor creation. Different hunks. |
| **#1353** (open) | Pretrained layer-weight initialization. Unchanged. |
| **#1369** (open) | Classifier-mode `predict`. Different path. |
| **#1400** (open) | `RBMRunner` in `rbm_digits.py`. Different file. Same NumPy-is-copied and same-device-identity rules used here. |
| **#1401** (open) | `MLPClassifier` inference. Different file. Same parent-`_apply` / meta / CUDA split. |
| **#1412** (open) | `Trainer` in `qvae_cell/trainer.py`. Different file. Same split. |
| **#1423** (open) | `FIDCalculator` in `qvae_mnist/utils/helpers.py`. Different file. This module follows its #164 matrix depth. |
| **#1393** (open) | Feature-selection classifiers. Different file. Same split; that PR does not cast caller dtype, while this path must follow the parameter dtype because `get_hidden` matmuls the batch. |
| **#331** (open) | Older `dbn_digits` fine-tune placement. Superseded in scope by #1378; no overlap with `get_layer_activations`. |

No other open PR edits `get_layer_activations` / `_batch_on_rbm_parameters` in `example/dbn_digits/supervised_dbn_digits.py`.

## Test plan

```bash
python3 -m unittest tests.test_dbn_layer_activation_parameter_device -v
python3 -m unittest tests.test_dbn_layer_activation_parameter_device tests.test_bm_device_sync -v
python3 -m black --check \
  example/dbn_digits/supervised_dbn_digits.py \
  tests/test_dbn_layer_activation_parameter_device.py
PYTHONPATH=src:example/dbn_digits python3 -m pylint --rcfile=pylintrc \
  tests/test_dbn_layer_activation_parameter_device.py
```

CPU-only VM, CPython 3.13.5, `torch==2.14.1+cpu`, black 26.10.1, pylint 4.1.3.

| Command | Result |
|---------|--------|
| `python3 -m unittest tests.test_dbn_layer_activation_parameter_device -v` | **23 tests, 17 passed, 6 skipped (CUDA)**, OK |
| `python3 -m unittest tests.test_dbn_layer_activation_parameter_device tests.test_bm_device_sync -v` | **28 tests, 20 passed, 8 skipped (CUDA)**, OK |
| `black --check` on both files | clean |
| `pylint --rcfile=pylintrc` on the test file | **10.00/10** |
| `pylint --rcfile=pylintrc` on the example | **7.92/10**, same score as the parent revision with the same command. Pre-existing example noise (`C0103`, `W0212`, `C0301`, ...) was not reformatted so it does not conflict with #1378. |

The 12 CPU sub-cases run on every machine; the 6 CUDA tests are collected and skipped without a GPU.

## Out of scope

- CUDA execution on this CPU-only VM (same gap as #164).
- Copying parameter values out of `meta` with `.to()` (that raises; the return trip uses `_apply` and `zeros_like`).
- `#1378` fine-tune batch placement, `#1353` weight init, `#1369` classifier-mode predict.
- `src/kaiwu/torch_plugin/dbn.py` (**#242** / **#1349**), RBM state propagation (**#1334**), RBM forward / Ising (**#1268**).
- Reformatting pre-existing source style.

## Commits (fork)

| SHA | Summary |
|-----|---------|
| `4febb03` | place layer activations on the live RBM parameter device (initial fix, 6-test module) |
| `ce6d791` | pass tensors through `Tensor.to` instead of NumPy |
| `2c8b9a2` | lock the #164 device matrix (meta identity, poisoned cache, `zeros_like` return, multi-layer) |
| branch head | sync this PR body |

## Merge-ready checklist

- [x] Fork branch `cursor/dbn-layer-activation-parameter-device-1269` pushed to TNHTH
- [x] Harden pass matches the #164 bar (CPU / `meta` executed; CUDA class present and skipped without a GPU)
- [ ] Maintainer review
