<!--
Coordinator: this file is the PR description. GitHub hides this comment.

gh pr edit 1401 --repo qboson/kaiwu-pytorch-plugin --body-file docs/pr-1401-body.md
-->

## Why

`MLPClassifier` in `example/qvae_mnist/downstream/classifier.py` sent `predict` / `predict_proba` batches to the construction-time `self.device` cache (`torch.FloatTensor(X).to(self.device)`). A parent `nn.Module._apply` moves the linear weights without calling the child `.to()` (merged **#164**, `tests/test_bm_device_sync.py`). After that move the batch and the weights diverge:

`RuntimeError: Tensor on device meta is not on the expected device cpu!`

Pointing the cache at `meta` while the weights stay on CPU is the opposite mix:

`RuntimeError: Tensor on device cpu is not on the expected device meta!`

Restoring cache placement (`torch.tensor(..., device=self.device)` into the linear stack after `parent._apply(meta)`, and the same helper swapped back onto `predict` / `predict_proba`) still raises that `RuntimeError`. The message contains both `meta` and `cpu`.

`torch.FloatTensor` on a float32 NumPy array aliases the caller buffer (`data_ptr` matches). An in-place write changes the caller's data. `np.asarray` on a `meta` tensor raises `TypeError` before the forward runs.

`Module.to(cpu)` cannot copy out of `meta` (`NotImplementedError: Cannot copy out of meta tensor; no data!`). Parameters stay on `meta`. The return trip retargets storage with `_apply` and zeros.

After `model.to(dtype=torch.float64)` or `parent.to(dtype=torch.float64)`, a float32 batch then dies inside the linear stack:

`RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Double`

On this torch (`2.7.0`), a forward whose weights and batch are both already on `meta` completes. `predict` / `predict_proba` then raise `NotImplementedError` from `.cpu()` (`Cannot copy out of meta tensor; no data!`). That is the fixed meta path. The device-mix `RuntimeError` is the pre-fix path.

## What

`self.device` stays the construction preference used by `fit` and `_create_model`. It is not turned into a live property. A later `fit` must not follow a module that a parent has already moved. Assignment does not move an attached network.

`predict` and `predict_proba` call `_features_on_parameter_device` on every call:

- The live parameter device is reread from the first parameter. With no parameters, the helper falls back to `self.device`.
- The live parameter dtype is reread the same way. With no parameters, the dtype is float32, matching `_create_model`.
- `Tensor.to` returns the same object when device and dtype already match, so same-device callers keep identity and `data_ptr`, including a noncontiguous view and a `meta` tensor. Meta values are not read.
- A different device or dtype is a copy. The caller is not written in place.
- NumPy is copied with `torch.tensor` onto that device and dtype. A float32 array is not aliased. A float64 NumPy array against float32 weights is still float32. Against float64 weights it is float64, which is what the linear op requires.

The linear head consumes activations that already sit on that device. These tests spy `Sequential` and, where the forward completes, the output `Linear`. Dropout stays in the module (`p=0.2`) and is inactive under `eval()`.

`tests/test_mlp_classifier_parameter_device.py` matches the #164 split and the #1400 / #1393 pattern. The CUDA class is behind `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")`.

**Every machine (CPU / `meta`):**

- An unmoved CPU classifier matches a same-weight reference on `predict` and `predict_proba`. Probabilities are finite and sum to one. The NumPy caller keeps storage, shape, dtype, layout, and values. `self.device` stays CPU.
- A noncontiguous float64 NumPy view is copied as float32 when the weights are float32, and the view is left intact.
- A float32 NumPy buffer is not aliased. `torch.FloatTensor` shares its `data_ptr`. Writing the tensor `predict` / `predict_proba` handed the network does not change the caller. `data_ptr` differs. Both methods are covered.
- A float32 CPU tensor, including a noncontiguous view, keeps object identity and `data_ptr` through both methods. The output linear sees a different tensor on CPU.
- A meta batch that already matches the weights keeps storage (`is` and `data_ptr`). Both methods raise `NotImplementedError` from `.cpu()`. The output linear sees meta activations, not the caller. The child `.to` is not called. Meta values are not read.
- A CPU tensor, including a noncontiguous view, keeps `data_ptr`, dtype, device, and values after `parent._apply(meta)`. The network observes a distinct meta tensor. Both methods raise `NotImplementedError`.
- `parent.to(meta)` and `parent.to(device=meta)` do not call the child `.to`. Every parameter lands on `meta`. There are no buffers. `self.device` stays CPU. `_parameter_device()` follows the weights.
- `parent._apply(meta)` does not call `.to`. A CPU NumPy `predict` and `predict_proba` place a float32 meta batch, then raise `NotImplementedError` (cannot copy the result to the host), not the meta/CPU `RuntimeError`. The caller is unchanged.
- Assigning `device` to `meta`, or to `cuda`, does not move CPU weights and does not change `predict` / `predict_proba`. Feeding a batch built on that meta cache into the linear stack raises `RuntimeError` naming both `meta` and `cpu`.
- Restoring the cache helper reproduces that `RuntimeError` on both methods. Deleting the override returns to the live-parameter path (`NotImplementedError`, meta batch).
- With no network, `_parameter_device()` follows `self.device`, including `meta`. An empty `Sequential` does the same. Real weights win over a cache pointed at the other device.
- A replacement `_create_model()` is built on the construction preference while the moved network stays on `meta`.
- `parent.to(dtype=torch.float64)` stays on CPU, does not call the child `.to`, and both methods return finite probabilities. The batch the network sees is float64. The NumPy caller is not aliased. Returned probabilities are float64.
- A direct `model.to(dtype=torch.float64)` is the one recorded `.to` call. It stays on CPU and both methods score the same way.
- A float64 tensor that matches float64 weights keeps identity through both methods.
- float64 weights survive a later `_apply(meta)` without recasting or rewriting the caller. A float32 CPU tensor is copied to float64 on `meta`. Both methods raise `NotImplementedError`. `.to` is not called.
- The helper is reread after meta and after the return to CPU. `parent.to(cpu)` raises `NotImplementedError` and leaves parameters on `meta`. The return trip retargets storage with `_apply` and zeros (the source tensor is not read). The CPU score after that return is finite even if `self.device` is set back to `meta`. A further `_apply(meta)` is reread again. The child `.to` is not called. The caller NumPy array is unchanged.

**CUDA** `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")` — collected; skipped on this CPU-only VM (same gap as #164):

- `parent.to("cuda")` accepts a CPU NumPy batch, does not call `.to`, and matches a same-weight reference. The NumPy snapshot is unchanged. `self.device` stays CPU.
- `parent._apply(cuda)` and `parent.to(device="cuda")` do not call `.to`. CPU NumPy scores match a CPU weight clone within `rtol=1e-4, atol=1e-5`.
- A CPU tensor keeps `data_ptr`. The network sees CUDA float32. Probabilities match the CPU weight clone.
- Forcing the construction preference back to CPU leaves CUDA parameters in charge. Both methods still match the live reference.
- After CUDA then CPU, a CUDA cache does not win. The child `.to` is not called.
- A batch that is already on CUDA, contiguous or not, keeps storage (`is` and `data_ptr`). Probabilities are finite float32.
- A CUDA batch is copied onto CPU parameters. Caller storage stays on CUDA. Both methods match the CPU batch.
- CUDA → CPU → CUDA is reread. `.to` is not called. The CPU stop matches the CPU weight clone exactly. The NumPy caller is unchanged.
- float64 weights survive a CUDA `_apply`. The CPU NumPy caller keeps its pointer and dtype. The batch the network sees is float64 on CUDA. Probabilities are float64 and match the CPU float64 clone.
- `parent.to(dtype=torch.float64)` after a CUDA `_apply` stays on CUDA, does not call the child `.to`, and a float64 CUDA batch keeps identity. Probabilities are finite float64.

## Dedupe

| Item | Relationship |
|------|----------------|
| **Merged #164** | Review bar: `tests/test_bm_device_sync.py` (parent move does not call child `.to`, `meta` on every machine, CUDA class behind `@skipUnless`, dtype-only move keeps the device). This PR follows that split for MLP inference. The live location is the parameter device, reread every call. `self.device` is the fit preference, not a second copy of `bm.device`. |
| **#1400** (open) | `RBMRunner.device` in `rbm_digits.py`. Different estimator. Same NumPy-is-copied and same-device-tensor rules used here. `device` there is a property because callers read it as the live device. This PR does not do that. |
| **#1393** (open) | TinyCNN / SimpleRNN / SimpleLSTM in `feature_selection_models.py`. Different file. Those forwards keep the caller dtype. This estimator copies NumPy onto the parameter dtype because the linear stack requires it. |
| **#1378** (open) | DBN fine-tune in `supervised_dbn_digits.py`. Different file. |
| **#1399** (open) | `tests/test_dbn_train_classifier.py`. Different test. No hunk in this MLP module. |
| **#1376 / #1375 / #1374–#1370 / #1349–#1343 / #1334 / #1268 / #1292…** (open) | Other modules. Unchanged. |
| **TNHTH #1 / #2 / #3 / #4** | Fork-only parks / controls. Upstream is this PR. |

No other open upstream PR edits this MLP `predict` / `predict_proba` path for parameter-device alignment.

## Validation

CPU-only VM, CPython 3.12.3, `torch==2.7.0+cpu`. CUDA tests were collected and skipped (`requires CUDA`). 33 tests collected in the MLP module (23 run, 10 skipped). Before `334b2d6`, seven of those tests failed: tensor identity (CPU, noncontiguous, float64, meta) and float32-vs-float64 after a dtype-only move.

| Command | Result |
|---------|--------|
| `python3 -m unittest tests.test_mlp_classifier_parameter_device -v` | **23 passed, 10 skipped** (CUDA) |
| `python3 -m pytest tests/test_mlp_classifier_parameter_device.py -o addopts= -q` | **23 passed, 10 skipped**, 16 subtests passed |
| `PYTHONPATH=src python3 -m unittest tests.test_mlp_classifier_parameter_device tests.test_bm_device_sync -v` | **26 passed, 12 skipped** (MLP 23 + 10 CUDA; #164 bar 3 passed + 2 CUDA skipped) |
| `python3 -m black --check tests/test_mlp_classifier_parameter_device.py` | clean (black 25.1.0) |
| `python3 -m black --check example/qvae_mnist/downstream/classifier.py` | pre-existing trailing whitespace and spacing only. The new helper is not in that diff. Not reformatted. |
| `python3 -m pylint --rcfile=pylintrc tests/test_mlp_classifier_parameter_device.py` | **10.00/10** (pylint 3.3.7) |
| `python3 -m pylint --rcfile=pylintrc example/qvae_mnist/downstream/classifier.py` | **8.24/10**. The parent revision of this file, same command, was **8.12/10**. Same message codes (`C0103`, `C0114`, `C0115`, `C0301`, `C0303`, `C0411`, `R0402`, `W0102`, `W0221`, `W1203`). No new codes. |

```bash
python3 -m unittest tests.test_mlp_classifier_parameter_device -v
python3 -m pytest tests/test_mlp_classifier_parameter_device.py -o addopts= -q
PYTHONPATH=src python3 -m unittest tests.test_mlp_classifier_parameter_device tests.test_bm_device_sync -v
python3 -m black --check tests/test_mlp_classifier_parameter_device.py
python3 -m black --check example/qvae_mnist/downstream/classifier.py
python3 -m pylint --rcfile=pylintrc tests/test_mlp_classifier_parameter_device.py
python3 -m pylint --rcfile=pylintrc example/qvae_mnist/downstream/classifier.py
```

## Out of scope

- CUDA class collected; not executed on this CPU-only VM (same gap as #164).
- Copying parameter *values* out of `meta` with `.to()` (that raises; the return trip uses `_apply` and zeros).
- Reformatting pre-existing black noise in `classifier.py` (trailing whitespace in `__init__` / `fit`).
- Training path beyond keeping `self.device` as the construction preference. `fit` still places the new network and its batches there.
- `score` beyond calling `predict`.
- `get_params`: this estimator already stores `batch_size_mlp` as `batch_size`. Not asserted.
- `rbm_digits` (#1400), `feature_selection_models` (#1393), `supervised_dbn_digits` (#1378), `FeatureExtractor` (#1343).

## Commits (fork)

| SHA | Summary |
|-----|---------|
| `73a9bcbb290452a9e1791e41f4fa3c3ebb695ac1` | place classifier inference on the live parameter device |
| `82dd843d1c3d237a6472a9d6a260ba996e0ecf08` | failing repro: tensor identity, NumPy alias, meta/CPU mix, parameter dtype |
| `334b2d65737a2701ec81ab6e4ca25365091bcd71` | keep aligned tensors and follow the parameter dtype |
| branch head | sync this PR body |

## Merge-ready checklist

- [x] Fork branch `cursor/mlp-classifier-parameter-device-14af` pushed to TNHTH
- [x] Harden pass matches the #164 bar (CPU / `meta` executed; CUDA class present and skipped without a GPU)
- [ ] Maintainer review
