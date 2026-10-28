<!--
Coordinator: this file is the PR description. GitHub hides this comment.

gh pr edit 1374 --repo qboson/kaiwu-pytorch-plugin --body-file docs/pr-1374-body.md
-->

## Why

`SimpleEncoder.hierarchical_posterior` in `example/qvae_mnist/model/networks.py` encoded `x` with the linear weights and then drew `rho = torch.rand(q.size())`. That factory always uses CPU and the default dtype. `smoothing_distribution.icdf(rho, q)` mixes the two. A parent `nn.Module.to` moves those weights through `_apply` and does not call `SimpleEncoder.to` (merged **#164**), so CPU `rho` hits an activation that already lives somewhere else:

`RuntimeError: Tensor on device meta is not on the expected device cpu!`

The same mix happens when `x` is already on the parameter device. The bug is the sample, not only the caller batch.

Open **#1294** locks hierarchy count and shapes on CPU. Its `icdf` stub returns `q` and drops `rho`, so it cannot see this path.

## What

- `SimpleEncoder._parameter_device(fallback)` — device of the first parameter, reread on every call. No cache. A parameter-free encoder returns `fallback` (`StopIteration` is not raised).
- `hierarchical_posterior`: `x = x.to(device)` and `rho = torch.rand(q.shape, device=device, dtype=q.dtype)` before `icdf`. The caller's tensor is not modified in place. Dtype is not recast.
- `tests/test_simple_encoder_parameter_device.py`, split the same way as `tests/test_bm_device_sync.py`.

No further production change. The harden pass adds the tests below. Forcing `rho` back to `torch.rand(q.size())` fails the meta and float64 cases with that `RuntimeError` (and with `float32 != float64` when the weights are float64).

**Every machine (CPU / `meta`):**

- An unmoved CPU encoder keeps the caller batch (`data_ptr` and values) and returns one finite sample per level inside `[0, 1]`. `rho` is CPU float32.
- `parent.to(meta)` does not call `SimpleEncoder.to`. Parameters land on `meta`.
- `parent._apply(meta)` accepts a noncontiguous CPU batch, does not call `SimpleEncoder.to`, and leaves the caller unchanged (`data_ptr`, device, dtype, values, still noncontiguous). `rho` and the samples are on `meta` with the caller's dtype.
- When the batch is already on `meta` with the weights, `rho` is still drawn on `meta` (it is not the caller tensor). Caller storage is unchanged. Meta values are not read.
- float64 weights survive `_apply(meta)` without recasting or rewriting the caller batch. `rho` and the samples stay float64 on `meta`.
- `parent.to(dtype=torch.float64)` stays on CPU, does not call the child `.to`, and the samples stay in `[0, 1]`.
- A direct `.to(dtype=torch.float64)` stays on CPU. The caller batch is unchanged and the samples stay in `[0, 1]`.
- The helper is reread after `meta` and after a later `_apply` back to CPU. `.to(cpu)` raises `NotImplementedError` (`Cannot copy out of meta tensor; no data!`) and leaves the module on `meta`, so the return trip retargets storage with `_apply`. A further `_apply(meta)` is reread again. `SimpleEncoder.to` is not called. The caller batch is unchanged across the trip. Samples after the empty-storage return are checked for device only: meta has no data to bring back.
- Empty encoders fall back to the supplied device. Moving the encoder with `_apply(meta)` wins over a CPU fallback.
- A parameter-free encoder follows a CPU batch (unit interval, shape equal to the batch) and a `meta` batch.
- Same-device alignment keeps autograd: `samples[0].sum().backward()` reaches the CPU batch and `weight`.

**CUDA** `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")` — collected; skipped on this CPU-only VM (same gap as #164):

- `parent._apply(cuda)` does not call `SimpleEncoder.to`. A CPU batch stays put (`data_ptr` and values). `rho` and the samples are finite on CUDA and inside `[0, 1]`.
- A batch that is already on CUDA keeps `rho` on CUDA.
- CUDA → CPU → CUDA is reread. `SimpleEncoder.to` is not called. The caller batch is unchanged.
- float64 survives a CUDA move. `rho` stays float64.
- `parent.to(dtype=torch.float64)` after a CUDA `_apply` stays on CUDA, does not call the child `.to`, and keeps the samples in `[0, 1]`.
- Backward from a CPU batch reaches CUDA `weight` and a CPU input gradient.
- A CUDA batch is scored on a CPU encoder. Caller storage stays on CUDA. Samples stay in `[0, 1]` on CPU.

**Not in scope:** #1294's hierarchy count, `encode` formula, and `decode` rejection. Library `QVAE.forward` (#1370). Cell `_encode` (#1371) and cell decoder reconstruction (#1372). MNIST feature extractor (#1343) and trainer placement (#1346). Copying parameter *values* out of `meta` with `.to()` (that raises; the return trip uses `_apply`). Other open device PRs' files.

## Dedupe

| Item | Relationship |
|------|----------------|
| **Merged #164** | Review bar: `tests/test_bm_device_sync.py` (parent move does not call child `.to`, `meta` on every machine, CUDA class behind `@skipUnless`, dtype-only move keeps the device). This PR follows that split for `SimpleEncoder.hierarchical_posterior`. The live location here is the encoder parameter, not `bm.device`. |
| **#1294** (open) | `tests/test_qvae_mnist_simple_encoder.py` only. CPU hierarchy count, sample shape, and `encode` / `decode`. Its `icdf` stub drops `rho`. This PR does not edit that file. Sample shape is asserted here only so the device checks are on the posterior tensor. The level count is read from `num_latent_hierarchy_levels` so every level's `rho` is on the parameter device. |
| **#825** (open) | `BasicDecoder.decode` when `activation_fct` is `None`, in the same `networks.py`. Different class. No overlap with `_parameter_device` or `hierarchical_posterior`. |
| **#430** (open) | `Decoder` (sequential network) init and `get_weight_decay`, same file. Different class. No overlap with the encoder posterior hunk. |
| **#1343 / #1346** (open) | Feature extractor and MNIST trainer / tuner device placement. Different files. |
| **#1370 / #1348** (open) | Library `qvae.py` reconstruction and BM-facing latents. This branch does not edit `src/kaiwu/torch_plugin/qvae.py`. |
| **#1371 / #1372** (open) | `example/qvae_cell/models.py`. No overlap. |
| **#1373** (open) | Example `DEQForecaster` embed device. Different files. |
| **#1344 / #1349 / #1347 / #1345 / #1334 / #1275 / #1268 / #1292 / #1291 / #1290** (open) | Library or other-example device work (QDEQ, DBN, full BM, Ising buffers, RBM, GBRBM, tensor factories, QDiffusion, `qvae_dist_util`, MAIFS). No `hierarchical_posterior` hunk. |
| **#488 / #415 / #163 / #485** (open) | Other device or `.to` contract work. Different files. |

If #825 or #430 lands on `networks.py`, keep both their decoder edits and this encoder device move. Do not drop `_parameter_device` or the `rho` device arguments to resolve a conflict. Do not rebase onto #1294, #1370, or #1371 to pre-merge them.

## Validation

On a CPU-only machine (torch 2.7.0+cpu, Python 3.12.3, no CUDA). Restoring `rho = torch.rand(q.size())` fails 7 tests (`RuntimeError: Tensor on device meta is not on the expected device cpu!`, plus float64 `rho` left as float32). With `rho` on the live parameter device:

| Command | Result |
|---------|--------|
| `PYTHONPATH=src python3 -m pytest tests/test_simple_encoder_parameter_device.py -o addopts='' -q` | **11 passed, 7 skipped** (CUDA class) |
| `python3 -m black --check tests/test_simple_encoder_parameter_device.py` | clean (black 25.1.0) |
| `PYTHONPATH=src python3 -m pylint --rcfile=pylintrc tests/test_simple_encoder_parameter_device.py` | **10.00/10** |
| same pylint, `--disable=C0114,C0116` | **10.00/10** |

```bash
PYTHONPATH=src python3 -m pytest tests/test_simple_encoder_parameter_device.py -o addopts='' -q
python3 -m black --check tests/test_simple_encoder_parameter_device.py
PYTHONPATH=src python3 -m pylint --rcfile=pylintrc tests/test_simple_encoder_parameter_device.py
PYTHONPATH=src python3 -m pylint tests/test_simple_encoder_parameter_device.py --disable=C0114,C0116
```

`black --check example/qvae_mnist/model/networks.py` still wants wraps and comment spacing that are already outside this diff (`BasicDecoder`, `Decoder`, and the file header). `_parameter_device` and the `hierarchical_posterior` lines are not part of that black diff, so this branch does not reformat the example. Reformatting it would collide with #825 and #430.

`pylint --rcfile=pylintrc example/qvae_mnist/model/networks.py` is **9.88/10**. The only findings are the pre-existing `__main__` block reading `_layers` (`protected-access`). They are not from the device change.

### Test matrix

| Test | Result here |
|------|-------------|
| `test_cpu_posterior_stays_in_unit_interval` | passed |
| `test_parent_to_meta_skips_encoder_to` | passed |
| `test_parent_apply_accepts_cpu_batch` | passed |
| `test_rho_follows_parameters_when_input_is_already_there` | passed |
| `test_float64_parameters_keep_feature_dtype_and_storage` | passed |
| `test_dtype_only_move_keeps_cpu_device` | passed |
| `test_direct_dtype_only_posterior_stays_in_unit_interval` | passed |
| `test_parameter_device_is_reread_after_meta_round_trip` | passed |
| `test_parameter_device_uses_live_parameters_or_fallback` | passed |
| `test_parameter_free_encoder_follows_the_input_device` | passed |
| `test_same_device_alignment_keeps_autograd` | passed |
| `test_parent_apply_accepts_cpu_batch` (CUDA class) | skipped (no CUDA) |
| `test_rho_follows_parameters_when_input_is_already_there` (CUDA class) | skipped (no CUDA) |
| `test_device_is_reread_across_cuda_and_cpu` | skipped (no CUDA) |
| `test_float64_cuda_move_keeps_dtype` | skipped (no CUDA) |
| `test_dtype_only_move_keeps_cuda_device` | skipped (no CUDA) |
| `test_backward_from_cpu_batch_reaches_cuda_weights` | skipped (no CUDA) |
| `test_cuda_batch_follows_cpu_parameters` | skipped (no CUDA) |

## Commits (fork)

| SHA | Summary |
|-----|---------|
| `4dc0c1d3d86147e0f5983fc8e6710d5f149514f6` | live parameter device for `x` / `rho` |
| `0a01d3137c98871e0798c47ba5904cceee7dad68` | #164-bar tests: caller `data_ptr` / values, dtype-only, meta round-trip, CUDA class |

## Merge-ready checklist

- [x] Fork branch pushed to TNHTH with local pytest / black / pylint
- [x] Harden pass to the #164 bar (CPU / `meta` matrix above)
- [ ] CUDA class executed on a GPU (collected and skipped here; 7 tests)
- [ ] Maintainer review

## Risk

Low. One non-in-place `.to` of the batch onto the live parameter device, and `rho` created on that device with `q.dtype`. No public API change. No edit under `src/kaiwu/torch_plugin/`. Head-less encoders follow the batch device.

Do not resolve a conflict with #825 or #430 by deleting `_parameter_device`. #1294 does not cover `rho`. #1370 and #1371 touch other encode paths.
