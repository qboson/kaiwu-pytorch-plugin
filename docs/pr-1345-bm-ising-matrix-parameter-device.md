# PR #1345 — `fix(bm): allocate Ising export buffers on the parameter device`

Upstream: https://github.com/Qboson/kaiwu-pytorch-plugin/pull/1345  
Fork branch: `cursor/bm-ising-matrix-parameter-device-8b0c`

Coordinator: if `ManagePullRequest` returns 403 on qboson, paste this body with:

```bash
gh pr edit 1345 --repo Qboson/kaiwu-pytorch-plugin --body-file docs/pr-1345-bm-ising-matrix-parameter-device.md
```

(Strip this header block and keep the section below the `---`.)

---

## Why

`BoltzmannMachine._to_ising_matrix()` still allocated Ising work buffers with cached `self.device`. After a parent `_apply` / `.to()` moves parameters (the **#164** scenario), `quadratic_coef` can sit on `meta`/CUDA while the buffer is created on CPU → `Cannot copy out of meta tensor` / mixed-device errors, so `get_ising_matrix()` / `sample()` cannot finish.

Open **#1268** already fixes `gibbs_sample` / `condition_sample` and RBM `_to_ising_matrix` paths; the **full** BM Ising export path was still stale.

## What

- `BoltzmannMachine._parameter_device()` — first parameter device, else `self.device`.
- `_to_ising_matrix` — allocate the work buffer with `device=self._parameter_device()`.
- `tests/test_bm_ising_matrix_parameter_device.py` — meta mechanism, explicit stale `self.device` repro (CPU buffer vs meta parameters), parent `_apply` without child `.to()`, `torch.zeros` device capture, dtype check, CUDA `@skipUnless` nested move + `sample()` integration.

**Not in scope:** other `full_boltzmann_machine.py` hunks already covered by **#1268**.

## Dedupe

| Item | Relationship |
|------|----------------|
| **#1268** | Same file, different path (`gibbs_sample` / `condition_sample` / `_hidden_to_ising_matrix`); complementary |
| **#1334** | RBM state propagation; different class |
| **Merged #164** | Pattern reference (`_apply`, parent moves, `tests/test_bm_device_sync.py`) |
| **#1275 / #1290–#1344** | No overlap |

## Validation

```bash
PYTHONPATH=src pytest tests/test_bm_ising_matrix_parameter_device.py -q
PYTHONPATH=src pytest tests/test_bm_device_sync.py tests/test_bm.py -q
PYTHONPATH=src pytest tests/ -q --ignore=tests/test_qdeq.py

python3 -m black --check \
  src/kaiwu/torch_plugin/full_boltzmann_machine.py \
  tests/test_bm_ising_matrix_parameter_device.py
PYTHONPATH=src python3 -m pylint \
  src/kaiwu/torch_plugin/full_boltzmann_machine.py \
  tests/test_bm_ising_matrix_parameter_device.py
```

## Risk

Low. Narrow allocation change in one method. Coordinate merge order with **#1268** (same file, different hunks).
