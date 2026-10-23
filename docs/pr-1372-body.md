<!--
Coordinator: this file is the PR description. GitHub hides this comment.

gh pr edit 1372 --repo qboson/kaiwu-pytorch-plugin --body-file docs/pr-1372-body.md
-->

## Why

`CellQVAE.forward` feeds encoder-side latents and the Bernoulli `_train_bias` buffer straight into the decoder. `CellQVAE` overrides `QVAE.forward` and does not call it, so library #1370 does not cover this path. A parent `nn.Module.to` moves the decoder through `_apply` and does not call `QVAEDecoder.to` (merged **#164**). CPU latents and `_train_bias` then hit decoder weights that have already moved (`RuntimeError` on `meta`, same class of crash as #164).

## What

- `CellQVAE._decoder_parameter_device(fallback)` — device of the first decoder parameter, reread on every call. It does not cache. A parameter-free decoder falls back to `fallback.device` (`StopIteration` is not raised).
- `forward`: copy `zeta` onto that device before `_decoder_input` / the decoder. Bernoulli `_train_bias` is added on `recon_x.device`. Caller `x`, the returned `zeta`, `batch_idx`, and the stored `_train_bias` are not modified in place. Dtype is unchanged. MSE does not add `_train_bias`.
- Does **not** change `_encode` (that is open #1371).
- `tests/test_cell_qvae_forward_parameter_device.py`, split the same way as `tests/test_bm_device_sync.py`. The encoder stays on the caller device, so these tests fail inside `forward` rather than `_encode`.

**Every machine (CPU / `meta`):**

- `parent.to(meta)` does not call `QVAEDecoder.to`.
- Decoder-only `_apply(meta)` accepts a CPU batch, does not call `decoder.to`, leaves the caller unchanged (`data_ptr` and values), leaves logits and the returned `zeta` on CPU, and puts `recon_x` on `meta`.
- Bernoulli CPU `_train_bias` stays on CPU (`data_ptr` and values) while `recon_x` is on `meta`.
- MSE ignores a `_train_bias` left on `meta`.
- Batch indices stay on CPU (`data_ptr` and values); the decoder input, including the one-hot, is on `meta`.
- float64 survives decoder `_apply(meta)`, including a batch one-hot.
- `.to(dtype=torch.float64)` stays on CPU; Bernoulli reconstruction matches a CPU weight copy.
- The helper is reread after decoder `meta` and after a later `_apply` back to CPU (`.to()` cannot copy out of `meta`). The encoder never leaves CPU.
- A meta encoder does not select the decoder device (helper only; `forward` is not called, because that crash is #1371's `_encode`).
- A parameter-free decoder follows the latent device for a CPU batch and a `meta` batch.
- Same-device MSE and Bernoulli-with-batch reconstructions match a CPU decoder copy. Caller tensors are unchanged.
- Same-device `.to` does not detach: `recon.sum().backward()` reaches CPU `decoder.fc1.weight` and a CPU input gradient.
- `forward` does not call `QVAE.forward` (#1370), including after a meta decoder move.

**CUDA** `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")`:

- Decoder-only `_apply(cuda)` does not call `decoder.to`. A CPU batch stays on CPU. Reconstruction matches a CPU decoder copy. `_train_bias` stays on CPU.
- Backward from a CPU batch reaches CUDA `decoder.fc1.weight` and a CPU input gradient. Returned `zeta` stays on CPU.
- Decoder CUDA → CPU → CUDA is reread. `decoder.to` is not called. The encoder stays on CPU. Reconstructions match a CPU weight copy.
- A CUDA latent (parameter-free encoder) reconstructs on a CPU decoder and matches a CPU weight copy.
- A CUDA `_train_bias` is added on a CPU decoder, the buffer stays on CUDA, and the reconstruction matches a CPU weight copy.
- Batch one-hot columns are on CUDA and match `one_hot`; `batch_idx` and `_train_bias` stay on CPU.
- float64 survives a CUDA decoder move and matches a CPU weight copy.

**Not in scope:** #1371's `_encoder_parameter_device` and the `_encode` input move. #1370's `QVAE.forward` / `loss`. #1348's BM-facing `energy` / `bm_loss` / `_cross_entropy`. #512's qvae_cell MPS selector. Whole-module `.cuda()` / `parent.to("cuda")` on `CellQVAE` (that also moves the encoder; #1371). Copying latents or `_train_bias` *out of* `meta` (`.to()` raises `NotImplementedError`; the return trip retargets decoder storage with `_apply`). `energy`'s pre-existing `(q > 0).float()` cast.

## Dedupe

| Item | Relationship |
|------|----------------|
| **#1371** (open) | Same file `example/qvae_cell/models.py`, **different methods**. #1371 adds `_encoder_parameter_device` and edits `_encode`. This PR adds `_decoder_parameter_device` and edits `forward` only. |
| **#1370** (open) | Library `qvae.py` `forward` / `loss`. `CellQVAE` does **not** call `super().forward()`. Locked by `test_forward_does_not_call_qvae_forward`. This branch does not edit `qvae.py`. |
| **#1348** (open) | Library BM-facing paths. This branch does not edit `CellQVAE.energy`. |
| **#512** (open) | `example/qvae_cell` runtime / train MPS selection. Different files. |
| **Merged #164** | Review bar: `tests/test_bm_device_sync.py` (parent move does not call child `.to`, `meta` on every machine, CUDA class behind `@skipUnless`, dtype-only move keeps the device, stale device must not win). This PR follows that split for `CellQVAE.forward`. The live location here is the decoder parameter, not `bm.device`. |

### Overlap with #1371 — helper placement

Both branches are based on `main` (`dca313e`). This branch does **not** contain #1371's hunks, and it does not rewrite them.

| Branch | Where the hunk sits |
|--------|---------------------|
| #1371 | Inserts `_encoder_parameter_device` immediately **before** `_encode`, plus `x = x.to(...)` inside `_encode` (`@@ -148` and `@@ -161` against `77da676`). |
| This PR | One hunk, `@@ -177,21 +177,44 @@`: inserts `_decoder_parameter_device` immediately **after** `_encode` and before `forward`, and edits `forward` only. |

`git merge-tree --write-tree` of this branch with `TNHTH:cursor/cell-qvae-encode-parameter-device-4e91` (tip `91113efc`, the current #1371 head) **auto-merges** `example/qvae_cell/models.py` (exit 0). Both helpers are present. There are no conflict markers against that tip.

The conflict risk is the placement, not a rewrite. The two helpers bracket `_encode`. A later edit that touches the tail of `_encode` (`return q, posterior, zeta`), the blank line before `forward`, or either docstring in that neighborhood will collide when the second PR merges.

**Resolution if markers appear:** keep **both** helpers and **both** call sites.

- Keep `_encoder_parameter_device` and `x = x.to(self._encoder_parameter_device(x))` in `_encode`.
- Keep `_decoder_parameter_device`, `decoder_zeta = zeta.to(...)`, and `self._train_bias.to(device=recon_x.device)` in `forward`.

Do not take "ours" or "theirs" for the whole file. Do not delete either helper to clear the conflict. Do not rebase this branch onto #1371 to pre-merge; the diff against `main` must stay the forward hunk only.

Test files do not overlap: `tests/test_cell_qvae_parameter_device.py` (#1371) and `tests/test_cell_qvae_forward_parameter_device.py` (this PR). They load `models.py` under different `sys.modules` keys (`qvae_cell_models` vs `qvae_cell_models_forward`).

No overlap with #1370 / #1348 inside `src/kaiwu/torch_plugin/qvae.py`.

## Validation

On a CPU-only machine (torch 2.7.0+cpu, Python 3.12.3, no CUDA). A helper that returned `fallback.device` would fail the meta decoder-move and reread tests: those tests require `recon_x` on `meta` while the caller batch, logits, and returned `zeta` stay on CPU. With the live decoder parameter device:

| Command | Result |
|---------|--------|
| `PYTHONPATH=src python3 -m pytest tests/test_cell_qvae_forward_parameter_device.py -o addopts='' -q` | 14 passed, 7 skipped |
| `python3 -m black --check` on `example/qvae_cell/models.py` and `tests/test_cell_qvae_forward_parameter_device.py` | clean (black 25.1.0) |
| `PYTHONPATH=src python3 -m pylint tests/test_cell_qvae_forward_parameter_device.py` | 10.00/10 (pylint 4.1.3) |
| same pylint, `--disable=C0114,C0116` | 10.00/10 |

```bash
PYTHONPATH=src python3 -m pytest tests/test_cell_qvae_forward_parameter_device.py -o addopts='' -q

python3 -m black --check \
  example/qvae_cell/models.py \
  tests/test_cell_qvae_forward_parameter_device.py
PYTHONPATH=src python3 -m pylint tests/test_cell_qvae_forward_parameter_device.py
PYTHONPATH=src python3 -m pylint tests/test_cell_qvae_forward_parameter_device.py --disable=C0114,C0116
```

The 7 skips are the CUDA class. They are collected and skipped unless `torch.cuda.is_available()`. They were not executed in the CPU-only run above. `example/qvae_cell/models.py` is unchanged in this hardening commit; black --check on it is clean. The forward hunk is still the one from `9794ab72`.

## Risk

Low. `forward` copies the latent and, for Bernoulli, the bias onto the decoder parameter device. No public API change. `_encode` and the BM visible-state device are not edited.

Do not resolve a conflict with #1371 by deleting `_encoder_parameter_device` or `_decoder_parameter_device`. #1370 and #1348 touch `qvae.py` only.

Supersedes tip `9794ab72` on `TNHTH:cursor/cell-qvae-forward-parameter-device-eb17`.
