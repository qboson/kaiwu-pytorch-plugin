<!--
Coordinator: this file is the PR description. GitHub hides this comment.

gh pr edit 1347 --repo qboson/kaiwu-pytorch-plugin --body-file docs/pr-1347-body.md
-->

## Why

Open [#1268](https://github.com/qboson/kaiwu-pytorch-plugin/pull/1268) already coerces inputs on **RBM** `forward` and on the sampling helpers (`sample`, `gibbs_sample`, `condition_sample`, `_hidden_to_ising_matrix`). The full **`BoltzmannMachine.forward`** Hamiltonian still mixed a caller tensor with parameters that live on another device.

That is the merged **#164** scenario, one step further: `nn.Module.to` moves a nested Boltzmann machine through `_apply` and does not call the child's `.to()` override. #164 keeps the plain `self.device` attribute in sync, but `forward` never moved `s_all`. A CPU batch against CUDA or `meta` parameters therefore raises a device-mismatch `RuntimeError`. The same crash remains when `self.device` is stale (parameter storage replaced without `_apply`) and only the live parameter device is correct.

`objective` calls `forward` twice, so the training entry has the same bug.

## What

- `AbstractBoltzmannMachine._model_device()` — device of the first parameter, otherwise `self.device`. Same helper body as #1268. This PR does not retarget `sample()` or any RBM path.
- `BoltzmannMachine.forward` — `s_all = s_all.to(self._model_device())` before the Hamiltonian. The caller's tensor is not modified in place. A stale `self.device` is not consulted.
- `tests/test_bm_forward_parameter_device.py`, split the same way as `tests/test_bm_device_sync.py`:
  - **Every machine (CPU / `meta`)**: `_model_device` fallback with no parameters; `_apply` move accepts a CPU batch and leaves that batch on CPU; parent `_apply` does not call the child `.to()`; stale `self.device` still scores on the parameter device; hand-checked Hamiltonian plus input and `linear_bias` gradients; float64 preserved across the move; `.to(dtype=...)` keeps CPU scoring; `objective` after `_apply`.
  - **CUDA** `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")`: construct-on-CUDA CPU batch matches a CPU weight reference; `parent.to("cpu")` accepts a CUDA batch; stale cache still scores on CUDA; `objective` lands on CUDA; backward from a CPU batch reaches CUDA parameters and the original CPU input.

**Not in scope:** #1268 sampling / RBM hunks, and #1345 Ising export buffers (`_parameter_device` / `_to_ising_matrix`).

## Dedupe

| Item | Relationship |
|------|----------------|
| **#1268** (open) | Adds the same `_model_device()` helper and uses it in `sample`, RBM `forward` / `_to_ising_matrix`, and full-BM `gibbs_sample` / `condition_sample` / `_hidden_to_ising_matrix`. **This PR does not repeat those hunks.** Overlap is only the helper on `abstract_boltzmann_machine.py` (identical body; docstring differs). Full-BM `forward` is not in #1268. |
| **#1345** (open) | `full_boltzmann_machine.py` Ising export buffer via `_parameter_device()` inside `_to_ising_matrix`. Different method, different hunk. This PR does not allocate Ising buffers. |
| **Merged #164** | Review bar: `tests/test_bm_device_sync.py` (parent `_apply`, meta mechanism on every machine, CUDA class behind `@skipUnless`). This PR follows that split for `forward` / `objective`. |

## Validation

Reverting `s_all.to(self._model_device())` fails the five meta-device forward / `objective` tests with `Tensor on device meta is not on the expected device cpu!`. Coercing to the cached `self.device` instead still fails `test_forward_follows_parameters_when_device_cache_is_stale`. With the parameter-device coercion, on a CPU-only machine (torch 2.7.0, no CUDA):

| Command | Result |
|---------|--------|
| `PYTHONPATH=src python3 -m pytest tests/test_bm_forward_parameter_device.py -v` | 8 passed, 5 skipped (CUDA class) |
| `PYTHONPATH=src python3 -m pytest tests/test_bm_device_sync.py -q` | 3 passed, 2 skipped (CUDA class) |
| `python3 -m black --check` on the two touched sources and the new test | clean |
| `PYTHONPATH=src python3 -m pylint tests/test_bm_forward_parameter_device.py` | 10.00/10 |
| `PYTHONPATH=src python3 -m pylint` on the two touched sources (`--disable=C0114,C0116`, same as CI) | 10.00/10 |

```bash
PYTHONPATH=src python3 -m pytest tests/test_bm_forward_parameter_device.py -v
PYTHONPATH=src python3 -m pytest tests/test_bm_device_sync.py -q

python3 -m black --check \
  src/kaiwu/torch_plugin/abstract_boltzmann_machine.py \
  src/kaiwu/torch_plugin/full_boltzmann_machine.py \
  tests/test_bm_forward_parameter_device.py
PYTHONPATH=src python3 -m pylint tests/test_bm_forward_parameter_device.py
```

CUDA cases are collected and skipped unless `torch.cuda.is_available()`. They were not executed in the CPU-only run above.

## Risk

Low. One coercion at the full-BM Hamiltonian entry. No public API change. Merge #1268 first or second: expect a docstring-only conflict on `_model_device`, not on `forward`. Do not combine with #1345's `_to_ising_matrix` edit; the hunks do not overlap.
