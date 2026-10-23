<!--
Coordinator: this file is the PR description. GitHub hides this comment.

gh pr edit 1370 --repo qboson/kaiwu-pytorch-plugin --body-file docs/pr-1370-body.md
-->

## Why

`QVAE.forward` and `loss` feed caller tensors straight into the encoder, the decoder, and the Bernoulli `_train_bias` buffer. Those three are separate pieces of module state. A parent `nn.Module.to` moves children through `_apply` and does not call their `.to()` (merged **#164**), so a CPU batch often meets parameters that have already moved. Moving only the decoder leaves `_train_bias` on its old device while `recon_x` is on the decoder device.

That is the same crash #164 fixed one level down: `RuntimeError: Tensor on device meta is not on the expected device cpu!`.

## What

- `QVAE._parameter_device(module, fallback)` — device of the first parameter of `module`, reread on every call. It does not cache. A parameter-free submodule falls back to `fallback.device` (`StopIteration` is not raised).
- `forward`: move `x` onto the encoder parameter device; move `zeta` onto the decoder parameter device before `decode`; add `_train_bias` on `recon_x.device`. The buffer itself is not moved, so a stale buffer device does not choose the reconstruction device. Caller tensors are not modified in place.
- `loss`: move `recon_x` and the targets onto the decoder parameter device before the MSE or Bernoulli reconstruction term. Caller tensors are not modified in place.
- `tests/test_qvae_reconstruction_device.py`, split the same way as `tests/test_bm_device_sync.py`:
  - **Every machine (CPU / `meta`)**: parent `_apply(meta)` accepts a CPU batch and does not call `QVAE.to` / `encoder.to` / `decoder.to`; direct `decoder.to(meta)` aligns latents and does not move the module again during `forward`; decoder-only `_apply` keeps `_train_bias` on CPU while `recon_x` is `meta`; MSE and Bernoulli `loss` deliver the reconstruction term on the decoder device (spied at `mse_loss` / `log_prob_per_var`, so a meta KL zero cannot hide a CPU term) and leave the caller tensors on CPU; same-device MSE and Bernoulli terms match the previous formulas; float64 survives `_apply(meta)`; `.to(dtype=torch.float64)` stays on CPU; the helper is reread after a later move back to CPU (`.to()` cannot copy out of `meta`, so the return trip uses `_apply`); a CPU dataset mean is applied on the encoder device; parameter-free encoder falls back to the batch device; parameter-free decoder falls back to the latent device; parameter-free `loss` falls back to `recon_x.device`; same-device alignment keeps autograd into the input and into encoder and decoder weights.
  - **CUDA** `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")`: `parent.to("cuda")` accepts a CPU batch, does not call child `.to()`, and matches a CPU weight copy; `parent.to("cuda")` then `parent.to("cpu")` accepts a CUDA batch; encoder and decoder constructed on CUDA keep the CPU `_train_bias` buffer and match a CPU weight copy; encoder-on-CUDA / decoder-on-CPU splits activations per submodule; `loss` scores CPU targets on the CUDA decoder and matches CPU MSE; backward from a CPU batch reaches CUDA encoder and decoder weights and a CPU input gradient; parent CUDA → CPU → CUDA is reread.

**Not in scope:** #1348's `_bm_parameter_device` and the `energy` / `bm_loss` / `_cross_entropy` edits. #1268's KL `entropy.to(cross_entropy.device)`. #1346's MNIST trainer placement. #1349's `dbn.py` activations. `example/qvae_cell` `CellQVAE.forward` (it overrides the base). Device-split `loss` tests replace `_kl_dist_from` and `_weight_decay_loss` because those terms follow the BM, not the decoder.

## Dedupe

| Item | Relationship |
|------|----------------|
| **#1348** (open) | Same file `qvae.py`, **different methods**. #1348 adds `_bm_parameter_device` and edits `energy` / `bm_loss` / `_cross_entropy`. This PR adds `_parameter_device` and edits `forward` / `loss` only. This branch is based on `main` and does **not** contain the #1348 hunks. Do not rebase onto #1348 to pre-merge them. |
| **#1268** (open) | KL `entropy.to(...)` only. This PR does not change `_kl_dist_from`. |
| **#1346** (open) | MNIST trainer batch placement. Does not edit `qvae.py`. |
| **#1349** (open) | `dbn.py` activations. Different module. |
| **Merged #164** | Review bar: `tests/test_bm_device_sync.py` (parent `_apply` does not call child `.to`, `meta` on every machine, CUDA class behind `@skipUnless`, dtype-only move keeps the device, stale device must not win). This PR follows that split for QVAE reconstruction. The stale location here is `_train_bias`, not `bm.device`. |

### Helper-placement conflict with #1348

Both PRs insert a new method in `src/kaiwu/torch_plugin/qvae.py` at the same anchor: after `self.create_networks()` in `QVAE.__init__`, and before `def create_networks`.

- #1348 inserts the instance method `_bm_parameter_device`.
- This PR inserts the static method `_parameter_device`.

Expect a trivial conflict on that insertion hunk when the second PR lands. The method bodies do not rewrite each other. Resolution: keep **both** helpers. Do not drop either one.

The other hunks should not overlap:

| Region | #1348 | this PR (#1370) |
|--------|-------|-----------------|
| `energy` / `bm_loss` / `_cross_entropy` | aligns latents to the BM parameter device | untouched |
| `forward` | untouched | `x.to(encoder)`, `zeta.to(decoder)`, `_train_bias.to(recon_x.device)` |
| `loss` | untouched | targets and `recon_x` moved to the decoder parameter device |
| `tests/test_qvae_bm_device.py` | added | not in this branch |
| `tests/test_qvae_reconstruction_device.py` | not in that branch | added |

## Validation

Forcing `_parameter_device()` to return `fallback.device` fails the meta parent-move, decoder-only, and `loss` tests with `RuntimeError: Tensor on device meta is not on the expected device cpu!` (forward) or with the spied reconstruction tensors staying on CPU (`loss`). With the live parameter device, on a CPU-only machine (torch 2.7.0+cpu, Python 3.12, no CUDA):

| Command | Result |
|---------|--------|
| `PYTHONPATH=src python3 -m pytest tests/test_qvae_reconstruction_device.py tests/test_qvae.py tests/test_bm_device_sync.py -q` | 26 passed, 9 skipped |
| `python3 -m black --check` on `src/kaiwu/torch_plugin/qvae.py` and `tests/test_qvae_reconstruction_device.py` | clean (black 25.1.0) |
| `PYTHONPATH=src python3 -m pylint tests/test_qvae_reconstruction_device.py` | 10.00/10 |
| `PYTHONPATH=src python3 -m pylint src/kaiwu/torch_plugin/qvae.py` | 10.00/10 |
| same pylint, `--disable=C0114,C0116` (CI flags) | 10.00/10 |

```bash
PYTHONPATH=src python3 -m pytest \
  tests/test_qvae_reconstruction_device.py \
  tests/test_qvae.py \
  tests/test_bm_device_sync.py \
  -q

python3 -m black --check \
  src/kaiwu/torch_plugin/qvae.py \
  tests/test_qvae_reconstruction_device.py
PYTHONPATH=src python3 -m pylint tests/test_qvae_reconstruction_device.py
PYTHONPATH=src python3 -m pylint src/kaiwu/torch_plugin/qvae.py
PYTHONPATH=src python3 -m pylint src/kaiwu/torch_plugin/qvae.py --disable=C0114,C0116
```

The 9 skips are the 7 CUDA tests in `tests/test_qvae_reconstruction_device.py` and the 2 CUDA tests in `tests/test_bm_device_sync.py` (that file is not modified). CUDA cases are collected and skipped unless `torch.cuda.is_available()`. They were not executed in the CPU-only run above.

`tests/test_qvae.py` (6) and the CPU class of `tests/test_bm_device_sync.py` (3) are the light regression next to the 17 CPU reconstruction tests.

## Risk

Low. Reconstruction copies activations onto the encoder or decoder parameter device. No public API change. `_train_bias` storage is left where it is.

Do not combine the helper insertion with #1348 by deleting one of the two methods. #1268's `_kl_dist_from` edit is adjacent in spirit and is not in this branch. #1346 and #1349 touch other files.

Supersedes tip `e35534ab` on `TNHTH:cursor/qvae-recon-parameter-device-7c2a`.
