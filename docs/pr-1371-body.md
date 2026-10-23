<!--
Coordinator: this file is the PR description. GitHub hides this comment.

gh pr edit 1371 --repo qboson/kaiwu-pytorch-plugin --body-file docs/pr-1371-body.md
-->

## Why

`CellQVAE` overrides `QVAE.forward` and `QVAE.energy` and encodes through `_encode`. It does not call `super().forward()` or `super().energy()`. A parent `nn.Module.to` moves children through `_apply` and does not call `CellQVAE.to` or `QVAEEncoder.to` (merged **#164**), so a CPU expression matrix still enters `fc1` against encoder parameters that have already moved.

That is the same crash #164 fixed one level down: `RuntimeError: Tensor on device meta is not on the expected device cpu!`.

## What

- `CellQVAE._encoder_parameter_device(fallback)` — device of the first encoder parameter, reread on every call. It does not cache. A parameter-free encoder falls back to `fallback.device` (`StopIteration` is not raised).
- `_encode`: move `x` onto that device before the encoder. Bernoulli dataset-mean subtraction uses the aligned tensor's device and dtype. Caller tensors are not modified in place. Dtype is unchanged.
- `tests/test_cell_qvae_parameter_device.py`, split the same way as `tests/test_bm_device_sync.py`:
  - **Every machine (CPU / `meta`)**: parent `_apply(meta)` accepts a CPU batch for `forward` and `energy` and does not mutate the caller (`data_ptr` and values); parent `_apply` does not call `CellQVAE.to` or `encoder.to`; Bernoulli CPU dataset mean is subtracted on the encoder device (spied at `encoder.forward`) and stays on CPU while the encoder input moves to `meta`; batch indices stay on CPU; float64 survives `_apply(meta)`; `.to(dtype=torch.float64)` stays on CPU and logits match a CPU weight copy; the helper is reread after a later move back to CPU (`.to()` cannot copy out of `meta`, so the return trip uses `_apply`), including `energy`; encoder-only `_apply(meta)` aligns `_encode` and does not call `encoder.to`, while the decoder stays on CPU; a meta decoder does not pull CPU encoder inputs off CPU; parameter-free encoder falls back to the batch device (CPU forward, and a `meta` batch); same-device logits match the encoder weights; same-device alignment keeps autograd into the input and `encoder.fc1.weight`; `forward` / `energy` do not call `QVAE.forward` / `QVAE.energy`.
  - **CUDA** `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")`: `.cuda()` accepts a CPU batch, leaves it on CPU, and encoder logits match a CPU weight copy; `parent.to("cpu")` after `.cuda()` accepts a CUDA batch, does not call `CellQVAE.to` again, and logits match; `parent.to("cuda")` does not call `CellQVAE.to` or `encoder.to` and logits match; backward from a CPU batch reaches CUDA `encoder.fc1.weight` and a CPU input gradient; parent CUDA → CPU → CUDA is reread; encoder-only CUDA `_encode` matches CPU logits while the decoder stays on CPU.

**Not in scope:** #1370's `_parameter_device` and `QVAE.forward` / `loss`. #1348's `_bm_parameter_device` and library `energy` / `bm_loss` / `_cross_entropy`. Decoder-only reconstruction on `cursor/cell-qvae-forward-parameter-device-eb17` (`_decoder_parameter_device`, `zeta.to(decoder)`, `_train_bias.to(recon_x.device)`). Encoder-only tests call `_encode` and do not claim `forward` succeeds when the decoder device differs. #512's qvae_cell MPS selector. `energy`'s pre-existing `(q > 0).float()` cast (float64 BM weights then raise `RuntimeError: expected scalar type Float but found Double`); that cast is the same contract as library `QVAE.energy` and is not a device choice.

## Dedupe

| Item | Relationship |
|------|----------------|
| **#1370** (open) | Library `qvae.py` `forward` / `loss`. `CellQVAE` does **not** call `super().forward()`. Locked by `test_forward_does_not_call_qvae_forward`. This branch does not edit `qvae.py`. |
| **#1348** (open) | Library BM-facing `energy` / `bm_loss` / `_cross_entropy`. `CellQVAE.energy` does **not** call `super().energy()`. Locked by `test_energy_does_not_call_qvae_energy`. This branch does not add `_bm_parameter_device`. |
| **#512** (open) | `example/qvae_cell` runtime / train MPS selection. Different files. |
| **Decoder sibling** `cursor/cell-qvae-forward-parameter-device-eb17` | Same file `example/qvae_cell/models.py`, **different methods**. That branch adds `_decoder_parameter_device` and edits `forward`. This branch adds `_encoder_parameter_device` and the `_encode` input move. |
| **Merged #164** | Review bar: `tests/test_bm_device_sync.py` (parent `_apply` does not call child `.to`, `meta` on every machine, CUDA class behind `@skipUnless`, dtype-only move keeps the device, stale device must not win). This PR follows that split for `CellQVAE._encode`. The live location here is the encoder parameter, not `bm.device`. |

### Overlap with the decoder sibling

Both branches edit `example/qvae_cell/models.py`.

- This PR inserts `_encoder_parameter_device` and one `x.to(...)` line in `_encode`.
- The decoder branch inserts `_decoder_parameter_device` and edits `forward` only.

Expect a small conflict on the `_encode` docstring / the line immediately above `forward` when the second PR lands. The method bodies do not rewrite each other. Resolution: keep **both** helpers, keep the `_encode` device move, and keep the decoder `forward` alignment. Do not drop either one.

No overlap with #1370 / #1348 inside `src/kaiwu/torch_plugin/qvae.py`. This branch does not contain those hunks. Do not rebase onto them to pre-merge.

## Validation

On a CPU-only machine (torch 2.7.0+cpu, Python 3.12, no CUDA), a helper that returned `fallback.device` would fail the meta parent-move, encoder-only, and reread tests: those tests require logits and reconstructions on `meta` while the caller batch stays on CPU. With the live parameter device:

| Command | Result |
|---------|--------|
| `PYTHONPATH=src python3 -m pytest tests/test_cell_qvae_parameter_device.py -o addopts='' -q` | 15 passed, 6 skipped |
| `python3 -m black --check` on `example/qvae_cell/models.py` and `tests/test_cell_qvae_parameter_device.py` | clean (black 25.1.0) |
| `PYTHONPATH=src python3 -m pylint tests/test_cell_qvae_parameter_device.py` | 10.00/10 |
| same pylint, `--disable=C0114,C0116` | 10.00/10 |

```bash
PYTHONPATH=src python3 -m pytest tests/test_cell_qvae_parameter_device.py -o addopts='' -q

python3 -m black --check \
  example/qvae_cell/models.py \
  tests/test_cell_qvae_parameter_device.py
PYTHONPATH=src python3 -m pylint tests/test_cell_qvae_parameter_device.py
PYTHONPATH=src python3 -m pylint tests/test_cell_qvae_parameter_device.py --disable=C0114,C0116
```

The 6 skips are the CUDA class in `tests/test_cell_qvae_parameter_device.py`. CUDA cases are collected and skipped unless `torch.cuda.is_available()`. They were not executed in the CPU-only run above. `example/qvae_cell/models.py` is unchanged in this hardening commit; black --check on it is clean.

## Risk

Low. `_encode` copies the input onto the encoder parameter device. No public API change. Decoder placement and the BM visible-state device are not edited.

Do not resolve a conflict with the decoder sibling by deleting `_encoder_parameter_device` or `_decoder_parameter_device`. #1370 and #1348 touch `qvae.py` only.

Supersedes tip `ae0155bc` on `TNHTH:cursor/cell-qvae-encode-parameter-device-4e91`.
