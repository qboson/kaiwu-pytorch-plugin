<!--
Coordinator: this file is the PR description. GitHub hides this comment.

gh pr edit 1375 --repo qboson/kaiwu-pytorch-plugin --body-file docs/pr-1375-body.md
-->

## Why

`BMConditionedEnergyModel` in `example/qdiffusion/dplm/models/bm.py` feeds caller tokens into the encoder, pools them, and runs the result through `feature_projector`. A parent `nn.Module.to` / `_apply` moves those parameters without calling `BMConditionedEnergyModel.to` (merged **#164**, `tests/test_bm_device_sync.py`). CPU tokens then hit weights that already live somewhere else:

`RuntimeError: Tensor on device cpu is not on the expected device meta!`

A projector-only move fails the other way when CPU features meet projector weights on `meta`. `energy_bm.device` is a plain attribute. After a parent move it can lag the parameters, and a Boltzmann machine constructed on CUDA starts with that cache on CUDA. It must not choose the device.

Restoring the pre-fix helpers (no `.to(device=...)` before `encode_tokens` / `feature_projector`) reproduces that `RuntimeError` on `parent._apply(meta)`.

## What

No further production change in the harden commit. `b85c702` already places features on the live parameter device:

- `_parameter_device(module, fallback)` — device of the first parameter, reread on every call. A parameter-free module returns `fallback.device` (`StopIteration` is not raised).
- `build_conditioned_features` aligns `noisy_tokens`, `candidate_tokens`, and `attention_mask` to the live encoder device. The caller's storage is not updated. Dtypes stay put. Same-device inputs stay the same objects (open **#1244** CPU identity).
- `build_visible_logits` aligns pooled features to the live projector device. The encoder and the projector may differ.
- Does **not** enter `score_visible_logits` / BM sampling.

`tests/test_bm_conditioned_energy_parameter_device.py` now matches the #164 split. The sampler raises if `solve` runs.

**Every machine (CPU / `meta`):**

- An unmoved CPU model reuses the caller objects (`is`), keeps noncontiguous storage (`data_ptr`, values, dtype, layout), and returns finite float32 logits of shape `(2, 3)`.
- `parent.to(meta)` and `parent.to(device=meta)` do not call `BMConditionedEnergyModel.to`. Encoder and projector parameters land on `meta`.
- `parent._apply(meta)` does not call `.to`. Caller tokens and the mask stay on CPU. The encoder observes `meta` copies. Logits are float32 on `meta`. Forcing `energy_bm.device` back to CPU afterwards still loses to the projector.
- A batch that is already on `meta` with the weights keeps its storage. The encoder sees those same objects. Meta values are not read.
- Projector-only `_apply(meta)` leaves the encoder and a CPU `energy_bm.device` alone. The encoder still sees the caller objects. Logits are on `meta`.
- Encoder-only `_apply(meta)` puts pooled features on `meta` with shape `(2, 8)` while the projector stays on CPU. `build_visible_logits` then raises `NotImplementedError` (`Cannot copy out of meta tensor; no data!`) instead of returning meta logits. Caller storage stays put.
- A stale `energy_bm.device = meta` on an unmoved CPU model does not move features. Logits stay finite on CPU, and the encoder still sees the caller objects.
- `parent.to(dtype=torch.float64)` stays on CPU, does not call the child `.to`, leaves caller tokens `long` and the mask `bool`, and returns finite float64 logits.
- A direct `.to(dtype=torch.float64)` stays on CPU. Caller storage is unchanged. Logits are finite float64.
- float64 weights survive a later `_apply(meta)` without recasting or rewriting the caller batch. Logits are float64 on `meta`.
- The helper is reread after CPU, after `meta`, and after a later `_apply` back to CPU. `.to(cpu)` raises `NotImplementedError` and leaves the module on `meta`, so the return trip retargets storage with `_apply`. A further `_apply(meta)` is reread again. `BMConditionedEnergyModel.to` is not called. Caller storage is unchanged across the trip. CPU logits after the zero-storage return are finite; meta results are checked for device and dtype only.
- Empty encoders fall back to the supplied tensor's device. Moving only the encoder wins over a CPU fallback; the projector is unchanged.
- A parameter-free encoder keeps CPU caller identity even when `energy_bm.device` is `meta`, then follows a projector `_apply(meta)` while that cache is forced back to CPU. A `meta` batch with the projector already on `meta` keeps caller storage.
- Same-device alignment keeps autograd: `logits.sum().backward()` reaches CPU `feature_projector.weight`, `bias`, and the encoder embedding. Caller `data_ptr`s stay put.

**CUDA** `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")` — collected; skipped on this CPU-only VM (same gap as #164):

- `parent._apply(cuda)` does not call `.to`. A CPU batch stays put. Forcing `energy_bm.device` back to CPU still yields finite CUDA logits that match a CPU weight copy.
- A batch that is already on CUDA keeps its storage, and the encoder sees those same objects.
- CUDA → CPU → CUDA is reread. `.to` is not called. The caller batch is unchanged. Each stop matches the CPU copy.
- float64 survives a CUDA move. Logits stay float64 and match the CPU float64 copy. Caller tokens stay `long`.
- `parent.to(dtype=torch.float64)` after a CUDA `_apply` stays on CUDA, does not call the child `.to`, and keeps float64 logits.
- Backward from a CPU batch reaches CUDA projector weight, bias, and the encoder embedding. Caller storage stays on CPU.
- A CUDA batch is scored on a CPU encoder and projector. Caller storage stays on CUDA. Logits match the CPU copy.
- Encoder-only CUDA: features follow the encoder, logits follow the CPU projector, and a stale `meta` `energy_bm.device` does not win.
- Projector-only CUDA: the encoder still sees the caller objects, logits are on CUDA, and a CPU `energy_bm.device` does not win.

**Not in scope:** library `qdiffusion.py` (#488 / #310, and the open docs/validation edits #1095 / #1117 / #1143 / #1157). Example generator token placement (#1292). `common.py` AMP pool (#259). BM sampling / `forward` / Ising buffers (#1268 / #1347 / #1345). RBM state propagation (#1334). Copying parameter *values* out of `meta` with `.to()` (that raises; the return trip uses `_apply`).

## Dedupe

| Item | Relationship |
|------|----------------|
| **Merged #164** | Review bar: `tests/test_bm_device_sync.py` (parent move does not call child `.to`, `meta` on every machine, CUDA class behind `@skipUnless`, dtype-only move keeps the device). This PR follows that split for `build_conditioned_features` / `build_visible_logits`. The live location here is the encoder or projector parameter, not `energy_bm.device`. |
| **#1244** (open) | `tests/test_dplm_bm_conditioned_features.py` only. CPU feature order and `encode_tokens` object identity. Its encoder is parameter-free, so the fallback plus same-device `.to` keeps that `is` check. This PR does not edit that file. |
| **#1292** (open) | Example places tokens on the generator's first parameter device (proposal path). Does not edit `models/bm.py`. |
| **#1268 / #1347** (open) | BM sampling / `BoltzmannMachine.forward`. This PR stops at the projector. The test sampler raises if `solve` runs. |
| **#1345** (open) | Ising export buffers in `full_boltzmann_machine.py`. Different file. |
| **#1334** (open) | RBM hidden-state propagation. Different file. |
| **#259** (open) | `common.py` mixed-precision pooling. Different file. |
| **#488 / #310** (open) | Library `qdiffusion.py` metadata / constructor dtype. Different file. |
| **#1374** (open) | MNIST `SimpleEncoder` posterior. Different file. Same #164 test shape, not the same bug. |

## Validation

CPU-only VM, CPython 3.12, `torch==2.7.0+cpu`. CUDA tests were collected and skipped.

| Command | Result |
|---------|--------|
| `python3 -m pytest tests/test_bm_conditioned_energy_parameter_device.py -o addopts='' -q` | **14 passed, 9 skipped** |
| `python3 -m black --check` on `bm.py` and the test | clean |
| `PYTHONPATH=src:example/qdiffusion python3 -m pylint` on `bm.py` and the test | **10.00/10** |

```bash
python3 -m pytest tests/test_bm_conditioned_energy_parameter_device.py -o addopts='' -q
python3 -m black --check example/qdiffusion/dplm/models/bm.py tests/test_bm_conditioned_energy_parameter_device.py
PYTHONPATH=src:example/qdiffusion python3 -m pylint tests/test_bm_conditioned_energy_parameter_device.py example/qdiffusion/dplm/models/bm.py
```

The test inserts `src` and `example/qdiffusion` on `sys.path`, so pytest does not need `PYTHONPATH`. Pylint does not execute that insert; the `PYTHONPATH` above is what makes its import check resolve.

## Commits (fork)

| SHA | Summary |
|-----|---------|
| `b85c70235367ba48af8476a88e7ebd727b808807` | live encoder/projector device |
| branch head | harden tests to the #164 bar; no production change |

## Merge-ready checklist

- [x] Fork branch `cursor/bm-energy-parameter-device-fb05` pushed to TNHTH
- [x] Harden pass matches the #164 bar (CPU / `meta` executed; CUDA class present and skipped without a GPU)
- [ ] Maintainer review
