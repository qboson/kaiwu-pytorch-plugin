# PR #1334 — `fix(rbm): propagate states on the parameter device after _apply moves`

Upstream: https://github.com/qboson/kaiwu-pytorch-plugin/pull/1334  
Fork branch: `cursor/rbm-state-propagation-parameter-device-64ee`

Coordinator: if `ManagePullRequest` returns 403 on qboson, paste this body with:

```bash
gh pr edit 1334 --repo qboson/kaiwu-pytorch-plugin --body-file docs/pr-1334-rbm-state-propagation-device.md
```

(Strip this header block and keep the section below the `---`.)

---

## Why

`RestrictedBoltzmannMachine.get_hidden` / `get_visible` allocated state buffers from `self.device` and used caller tensors in matmuls against the RBM parameters. After a parent module relocates weights through `nn.Module._apply` (without invoking the child's `.to()` override), parameters can live on `meta`/CUDA while callers still pass CPU states. The old path then mixed devices in buffer assignment or `matmul`, raising:

`RuntimeError: Tensor on device meta is not on the expected device cpu!`

This matches the class of bug fixed in merged **#164** (`AbstractBoltzmannMachine._apply` device sync + `tests/test_bm_device_sync.py`), but on the **state-propagation** API rather than energy / `gibbs_sample`.

## What

- `RestrictedBoltzmannMachine._parameter_device()` — first parameter device, else `self.device`.
- `get_hidden` / `get_visible` — coerce inputs and allocate buffers on `_parameter_device()` (and parameter dtype).
- `tests/test_rbm_state_propagation_parameter_device.py` — meta mechanism, explicit stale `self.device` repro, parent `_apply` without child `.to()`, Bernoulli + `forward` integration, CUDA `@skipUnless` nested move tests (aligned with `test_bm_device_sync.py`).

**Not in scope:** `forward` / `_to_ising_matrix` / `gibbs_sample` placement (**#1268**).

## Dedupe

| Item | Relationship |
|------|----------------|
| **#1268** | Same file, different hunks (`forward` / `_to_ising_matrix`); complementary |
| **#577 / #309** | Dtype / input normalization; still use `self.device` for placement |
| **Merged #164** | Pattern reference (`_apply`, parent moves, meta tests) |
| **#1275 / #1290 / #1291 / #1292 / #242** | No overlap |

## Validation

```bash
PYTHONPATH=src pytest tests/test_rbm_state_propagation_parameter_device.py -q
PYTHONPATH=src pytest tests/test_bm_device_sync.py tests/test_rbm.py -q
PYTHONPATH=src pytest tests/ -q --ignore=tests/test_qdeq.py

python3 -m black --check \
  src/kaiwu/torch_plugin/restricted_boltzmann_machine.py \
  tests/test_rbm_state_propagation_parameter_device.py
PYTHONPATH=src python3 -m pylint \
  src/kaiwu/torch_plugin/restricted_boltzmann_machine.py \
  tests/test_rbm_state_propagation_parameter_device.py
```

## Risk

Low. Narrow change to two methods; CPU-only behavior unchanged when parameters and `self.device` already agree.
