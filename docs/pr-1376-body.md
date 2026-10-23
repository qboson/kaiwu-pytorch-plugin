<!--
Coordinator: this file is the PR description. GitHub hides this comment.

gh pr edit 1376 --repo qboson/kaiwu-pytorch-plugin --body-file docs/pr-1376-body.md
-->

## Why

`_ModifiedEsmModel.forward` in `example/qdiffusion/dplm/models/esm_patch.py` built the attention mask on the **caller** tensor device, then handed it to `EsmEmbeddings`. After a parent `_apply` moves word and position embedding weights without calling a child `.to()` (merged **#164**, `tests/test_bm_device_sync.py`), a CPU mask is multiplied into activations that already live somewhere else:

`RuntimeError: Tensor on device meta is not on the expected device cpu!`

The same path adds caller `inputs_embeds` to the position-embedding parameters. Two further gaps show up once the mask itself is aligned:

- With `token_dropout=True` (the ESM masked-LM setting), `EsmEmbeddings` compares `input_ids` with those activations. CPU ids still raise the meta/CPU `RuntimeError` after `parent._apply(meta)`. Meta embedding weights happen to accept a foreign index and return an empty meta tensor, which is why a `token_dropout=False` forward can look green without moving the ids.
- `nn.Embedding` on a device that has data (CPU or CUDA) requires the index on the weight device. A CUDA parent move therefore rejects CPU token ids and CPU position ids even when the mask was copied.

Restoring the pre-fix helper (mask left on the caller device, ids not moved) reproduces that `RuntimeError` on `parent._apply(meta)`.

## What

`_align_embedding_activations` reads the live embedding parameter device on every call:

- `_parameter_device(module, fallback)` — device of the first parameter. A parameter-free module returns `fallback.device` (`StopIteration` is not raised).
- `_on_parameter_device` — `Tensor.to(device=...)` only. Same-device inputs stay the same objects. Dtypes stay put. Callers are not written in place.
- The default mask, a provided `attention_mask`, `inputs_embeds`, `input_ids`, and `position_ids` all follow that device before `EsmEmbeddings`. A missing mask is `float32` of shape `(batch, seq + past)`.
- Does **not** consult a cached device attribute, and does **not** enter library `QDiffusion`.

`tests/test_esm_mask_parameter_device.py` matches the #164 split. The CUDA class is behind `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")`.

**Every machine (CPU / `meta`):**

- An unmoved CPU model reuses the caller objects (`is`) for token ids, the bool mask, fused embeds, and position ids. Noncontiguous storage stays put (`data_ptr`, values, dtype, layout). Hidden states are finite float32 of shape `(2, 5, 16)`.
- `parent.to(meta)` and `parent.to(device=meta)` do not call `_ModifiedEsmModel.to`. Word and position weights land on `meta`.
- `parent._apply(meta)` does not call `.to`. CPU token ids stay on CPU (`long`, same `data_ptr`). The embedding module observes `meta` ids and a generated float32 mask of shape `(2, 5)`. Hidden states are float32 on `meta`. Covered with `token_dropout` off and on.
- An explicit CPU bool mask and explicit CPU position ids stay on CPU. The embedding module observes `meta` copies (`is not`). Caller `data_ptr`s are unchanged.
- CPU `inputs_embeds` stay on CPU. The embedding module observes `meta` float32 copies. Hidden states are float32 on `meta`.
- A batch that is already on `meta` with the weights keeps its storage. The embedding module sees those same objects. Meta values are not read.
- `parent.to(dtype=torch.float64)` stays on CPU, does not call the child `.to`, leaves caller tokens `long` and the mask `bool`, and returns finite float64 hidden states.
- A direct `.to(dtype=torch.float64)` stays on CPU. Caller storage is unchanged. Hidden states are finite float64.
- float64 weights survive a later `_apply(meta)` without recasting or rewriting the caller batch. Hidden states are float64 on `meta`. A float64 mask copied onto `meta` stays float64.
- The helper is reread after CPU, after `meta`, and after a later `_apply` back to CPU. `.to(cpu)` raises `NotImplementedError` and leaves the module on `meta`, so the return trip retargets storage with `_apply`. A further `_apply(meta)` is reread again. The child `.to` is not called. Caller storage is unchanged across the trip. CPU hidden states after the zero-storage return are finite; meta results are checked for device and dtype only.
- Empty modules fall back to the supplied tensor's device (`cpu` and `meta`). Moving only the embedding module wins over a CPU fallback; the encoder is unchanged. A missing mask uses the past length (`(2, 9)` when `past=4`).
- Same-device alignment keeps autograd: `hidden.sum().backward()` reaches CPU `position_embeddings.weight` and the fused-embed input. Caller `data_ptr`s stay put. Gradients are finite and nonzero.

**CUDA** `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")` — collected; skipped on this CPU-only VM (same gap as #164):

- `parent.to("cuda")` does not call `.to`. CPU token ids, a CPU mask, and CPU position ids stay put. Hidden states are finite float32 on CUDA and match a CPU weight copy.
- `parent._apply(cuda)` does the same for CPU `inputs_embeds` and a CPU mask. The embedding module observes CUDA copies. Hidden states match the CPU copy.
- A batch that is already on CUDA keeps its storage, and the embedding module sees those same objects. Hidden states match the CPU copy of that batch.
- CUDA → CPU → CUDA is reread. `.to` is not called. The caller batch is unchanged. Each stop matches the CPU copy. The CPU stop sees the caller objects.
- float64 survives a CUDA move. Hidden states stay float64 and match the CPU float64 copy. Caller tokens stay `long`.
- `parent.to(dtype=torch.float64)` after a CUDA `_apply` stays on CUDA, does not call the child `.to`, and keeps float64 hidden states.
- Backward from a CPU fused embed reaches the CUDA position-embedding weight. Caller storage stays on CPU. The input gradient stays on CPU.
- A CUDA batch is embedded on CPU parameters. Caller storage stays on CUDA. Hidden states match the CPU copy.

**Not in scope:** `bm.py` encoder/projector (#1375). `common.py` pool (#259). Example generator token placement (#1292). Library `qdiffusion.py` dtype/metadata (#310 / #488). Decoder cross-attention (`encoder_hidden_states` / `encoder_attention_mask`) — this backbone is not a decoder, and those tensors still follow the caller device. Copying parameter *values* out of `meta` with `.to()` (that raises; the return trip uses `_apply`).

## Dedupe

| Item | Relationship |
|------|----------------|
| **Merged #164** | Review bar: `tests/test_bm_device_sync.py` (parent move does not call child `.to`, `meta` on every machine, CUDA class behind `@skipUnless`, dtype-only move keeps the device). This PR follows that split for ESM embedding inputs. The live location is the embedding parameter, reread every call. |
| **#1375** (open) | `models/bm.py` conditioned energy encoder/projector. Different file. Does not edit `esm_patch.py`. |
| **#259** (open) | `common.py` `masked_mean_pool` precision. Different file. |
| **#1292** (open) | Places tokens on the example `QDiffusion` generator before the call. Does not enter `EsmEmbeddings`. |
| **#310 / #488** (open) | Library `qdiffusion.py` constructor dtype and move metadata. Different file. |
| **#1370 / #1371 / #1372 / #1373 / #1374** (open) | QVAE reconstruction, cell encode/decode, DEQ forecaster history, MNIST posterior. Other modules, same #164 test shape. |
| **#1268 / #1345 / #1347 / #1349 / #1334** (open) | BM / DBN / RBM parameter-device buffers and forwards. Different files. |

No other open PR edits `example/qdiffusion/dplm/models/esm_patch.py`.

## Validation

CPU-only VM, CPython 3.12.3, `torch==2.7.0+cpu`, `transformers==4.39.2`. CUDA tests were collected and skipped.

| Command | Result |
|---------|--------|
| `PYTHONPATH=src python3 -m pytest tests/test_esm_mask_parameter_device.py -o addopts='' -q` | **14 passed, 8 skipped** (CUDA) |
| `python3 -m black --check` on `esm_patch.py` and the test | clean |
| `PYTHONPATH=src python3 -m pylint` on the test | **10.00/10** |
| `PYTHONPATH=src python3 -m pylint` on `esm_patch.py` | **9.84/10** — three pre-existing warnings (`import-outside-toplevel` in `_ModifiedEsmModel.__init__`, missing docstrings on the two `forward`s). Same three warnings on `dc24400`. |

```bash
PYTHONPATH=src python3 -m pytest tests/test_esm_mask_parameter_device.py -o addopts='' -q
python3 -m black --check example/qdiffusion/dplm/models/esm_patch.py tests/test_esm_mask_parameter_device.py
PYTHONPATH=src python3 -m pylint tests/test_esm_mask_parameter_device.py
PYTHONPATH=src python3 -m pylint example/qdiffusion/dplm/models/esm_patch.py
```

## Commits (fork)

| SHA | Summary |
|-----|---------|
| `dc24400b9fbe3d2fd0222d2d0927d1ae6871b372` | align mask / fused embeds to the live embedding device |
| branch head | align token ids and position ids; harden tests to the #164 bar |

## Merge-ready checklist

- [x] Fork branch `cursor/esm-mask-parameter-device-c4e1` pushed to TNHTH
- [x] Harden pass matches the #164 bar (CPU / `meta` executed; CUDA class present and skipped without a GPU)
- [ ] Maintainer review
