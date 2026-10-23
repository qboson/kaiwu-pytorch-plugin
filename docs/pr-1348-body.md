<!--
Coordinator: this file is the PR description. GitHub hides this comment.

gh pr edit 1348 --repo qboson/kaiwu-pytorch-plugin --body-file docs/pr-1348-body.md
-->

## Why

QVAE keeps the encoder, decoder, and BM as **sibling** submodules. Moving only `qvae.bm` (two-phase BM fine-tuning, or a parent `nn.Module.to` that never calls `BM.to`) left `energy()`, `bm_loss()`, and KL `_cross_entropy()` feeding **encoder-device latents** into the BM.

That is the merged **#164** scenario one level up. #164 keeps a nested Boltzmann machine's plain `self.device` in sync inside `_apply`, but QVAE was still choosing the device of the encoder tensor. A CPU batch against `meta` or CUDA parameters raises `RuntimeError: Tensor on device meta is not on the expected device cpu!`. The same crash remains when `bm.device` is stale and only the live parameter device is correct.

## What

- `QVAE._bm_parameter_device()` — device of the first BM parameter, reread on every call. It does not cache and it does not consult `bm.device`. A parameter-free BM falls back to another QVAE parameter, then CPU (`StopIteration` is not raised).
- `energy`, `bm_loss`, and `_cross_entropy` align the visible state / `sigmoid(q)` / `logit_q` to that device before the BM call. Caller tensors are not modified in place. `energy()` still finishes the binary visible state with `.float()`, which is the previous contract.
- `tests/test_qvae_bm_device.py`, split the same way as `tests/test_bm_device_sync.py`:
  - **Every machine (CPU / `meta`)**: direct `bm.to(meta)` for `energy`, `bm_loss` (including weight decay), and `_cross_entropy`; parent `.to(meta)` does not call `BM.to` and still scores, with encoder outputs left on the batch device; `_apply` move plus a manually stale `bm.device`; the helper is reread after a later move back to CPU (`.to()` cannot copy out of `meta`, so the return trip uses `_apply`); `.to(dtype=...)` keeps CPU; parameter-free fallback to a sibling parameter and to CPU; same-device cross-entropy keeps autograd into the logits and BM weights.
  - **CUDA** `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")`: construct-on-CUDA CPU batch matches a CPU weight reference for `energy`, and `bm_loss` / `_cross_entropy` land on CUDA; `parent.to("cpu")` accepts a CUDA batch and does not call `BM.to`; parent round-trip CPU then CUDA is reread; stale `bm.device` still scores on CUDA; backward from CPU logits reaches CUDA `quadratic_coef` and `linear_bias`.

**Not in scope:** #1268's KL `entropy.to(cross_entropy.device)` hunk, #1268's `sample()` / RBM `_model_device()` retarget, #1347's full-BM `forward` coercion, and #1346's MNIST trainer placement. Stale-cache loss tests supply the negative phase on the parameter device because `sample()` still follows `bm.device`.

## Dedupe

| Item | Relationship |
|------|----------------|
| **#1268** (open) | RBM helpers plus KL `entropy.to(...)` in `QVAE._kl_dist_from`. **This PR does not add that line** and does not edit `abstract_boltzmann_machine.py`, `restricted_boltzmann_machine.py`, or `full_boltzmann_machine.py`. It aligns the three QVAE call sites that pass latents into the BM. |
| **#1347** (open) | Full `BoltzmannMachine.forward` input coerce and `tests/test_bm_forward_parameter_device.py`. Different type, different file. This PR does not touch those hunks. |
| **#1346** (open) | MNIST `ModelTuner` / trainer placement and `tests/test_mnist_device_placement.py`. Example loop, not `qvae.py`. |
| **Merged #164** | Review bar: `tests/test_bm_device_sync.py` (parent `_apply` does not call child `.to`, meta mechanism on every machine, CUDA class behind `@skipUnless`, stale device must not win). This PR follows that split for QVAE `energy` / `bm_loss` / `_cross_entropy`. |

## Validation

Forcing `_bm_parameter_device()` to stay on CPU fails the meta `energy`, parent-move, stale-cache, and `bm_loss` tests with `RuntimeError: Tensor on device meta is not on the expected device cpu!`. With the live parameter device, on a CPU-only machine (torch 2.7.0+cpu, Python 3.12, no CUDA):

| Command | Result |
|---------|--------|
| `PYTHONPATH=src python3 -m pytest tests/test_qvae_bm_device.py tests/test_qvae.py -q` | 19 passed, 5 skipped (CUDA class) |
| `PYTHONPATH=src python3 -m pytest tests/test_bm_device_sync.py -q` | 3 passed, 2 skipped (CUDA class; file not modified) |
| `python3 -m black --check` on `src/kaiwu/torch_plugin/qvae.py` and `tests/test_qvae_bm_device.py` | clean (black 25.1.0) |
| `PYTHONPATH=src python3 -m pylint tests/test_qvae_bm_device.py` | 10.00/10 |
| `PYTHONPATH=src python3 -m pylint src/kaiwu/torch_plugin/qvae.py` | 10.00/10 |
| same pylint, `--disable=C0114,C0116` (CI flags) | 10.00/10 |

```bash
PYTHONPATH=src python3 -m pytest tests/test_qvae_bm_device.py tests/test_qvae.py -q
PYTHONPATH=src python3 -m pytest tests/test_bm_device_sync.py -q

python3 -m black --check \
  src/kaiwu/torch_plugin/qvae.py \
  tests/test_qvae_bm_device.py
PYTHONPATH=src python3 -m pylint tests/test_qvae_bm_device.py
PYTHONPATH=src python3 -m pylint src/kaiwu/torch_plugin/qvae.py --disable=C0114,C0116
```

CUDA cases are collected and skipped unless `torch.cuda.is_available()`. They were not executed in the CPU-only run above.

## Risk

Low. Three QVAE call sites copy latents onto the BM parameter device before the prior runs. No public API change. Do not combine with #1268's `_kl_dist_from` edit: the hunks are adjacent but this branch does not contain `entropy.to(...)`. #1346 and #1347 touch other files and do not overlap.

Supersedes tip `6e9ebd78` on `TNHTH:cursor/qvae-bm-input-device-5bb5`.
