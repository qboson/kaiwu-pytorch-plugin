<!--
Coordinator: this file is the PR description. GitHub hides this comment.

gh pr edit 1378 --repo qboson/kaiwu-pytorch-plugin --body-file docs/pr-1378-body.md
-->

## Why

`SupervisedDBNClassification` in `example/dbn_digits/supervised_dbn_digits.py` placed fine-tune batches on `unsupervised_dbn.device`. That value is a plain attribute on the DBN. A parent `nn.Module._apply` moves `fine_tune_network` (`nn.Linear`) weights without calling `fine_tune_network.to` (merged **#164**, `tests/test_bm_device_sync.py`). Predict, probability, and backprop then mix devices:

`RuntimeError: Tensor on device meta is not on the expected device cpu!`

The same mix happens when the cache is forced to `meta` and the linear layer is still on CPU: the cache must not choose the device.

Class indices are the other tensor on that path. `CrossEntropyLoss` rejects a CPU target against meta logits with the same `RuntimeError`, including when `dropout_p > 0`. This is stricter than the #1376 embedding case: meta `nn.Embedding` can accept a foreign index, but meta cross-entropy does not. `int32` indices fail even on CPU (`expected scalar type Long but found Int`) unless they are cast to `long`. Dropout does not add a second caller index; once the features are on the parameter device, the dropout mask is generated there.

Restoring cache placement (features sent to `unsupervised_dbn.device`) reproduces the meta/CPU `RuntimeError` on `parent._apply(meta)`.

## What

`_align_fine_tune_features` / `_align_fine_tune_labels` read the **live** `fine_tune_network` parameter device and dtype on every call:

- With parameters, `unsupervised_dbn.device` is not consulted. A parameter-free module follows a tensor input; NumPy falls back to the DBN cache when that cache is a `torch.device`, otherwise CPU.
- `Tensor.to` keeps the same object when device and dtype already match, so same-device callers keep `data_ptr`. A different device is a copy. NumPy is copied. Callers are not written in place.
- Class indices are placed on the feature device as `torch.long`. An `int32` tensor is copied; a `long` tensor already on that device keeps its storage, including a noncontiguous view.
- Used from `_train_fine_tune_network`, `_predict_with_fine_tuning`, and `_predict_proba_fine_tuning`.

`tests/test_dbn_finetune_parameter_device.py` matches the #164 split. The CUDA class is behind `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")`.

**Every machine (CPU / `meta`):**

- An unmoved CPU network matches `argmax` / softmax of a CPU weight copy. NumPy inputs are not mutated.
- A meta DBN cache does not pull a CPU network off CPU. Predict and `predict_proba` still match the CPU copy.
- `parent.to(meta)` and `parent.to(device=meta)` do not call `fine_tune_network.to`. Every parameter lands on `meta`. The DBN cache stays CPU.
- `parent._apply(meta)` does not call `.to`. CPU NumPy predict and `predict_proba` raise `NotImplementedError` (cannot materialize meta), not the meta/CPU `RuntimeError`. Caller NumPy is unchanged.
- A CPU feature tensor keeps `data_ptr`, dtype, device, and values. The linear layer observes a distinct meta copy. A noncontiguous CPU view stays noncontiguous on CPU; the copy is meta.
- Class indices: a same-device `long` tensor, including a noncontiguous view, keeps storage. `int32` is copied to `long` and one CPU training step stays finite while the caller stays `int32`. After `parent._apply(meta)`, long, `int32`, NumPy, and `dropout_p=0.5` training all reach `Tensor.item() cannot be called on meta tensors`. The loss sees meta `float32` features, meta `long` indices, and meta logits. Caller storage stays on CPU. `.to` is not called. Forcing the cache back to CPU does not pull the step off meta.
- A meta batch that already matches the weights keeps feature and label storage. The forward stays `float32` on `meta`. Meta values are not read.
- `parent.to(dtype=torch.float64)` stays on CPU, does not call the child `.to`, and scores finite float64 logits. Caller `long` labels stay the same object.
- A direct `.to(dtype=torch.float64)` stays on CPU and scores finite float64 logits.
- float64 weights survive a later `_apply(meta)` without recasting or rewriting the caller batch. Logits are float64 on `meta`. Labels copied onto `meta` stay `long`. The caller stays float32 / long on CPU.
- The helper is reread after meta and after the return to CPU. `parent.to(cpu)` raises `NotImplementedError` and leaves every parameter on `meta`, so the return trip retargets storage with `_apply` and zeros (the source tensor is not read). A further `_apply(meta)` is reread again. The child `.to` is not called. The CPU score after the zero-storage return is finite.
- Empty modules follow a CPU tensor and a meta tensor (`is`, `data_ptr`) instead of the opposite cache. NumPy with no parameters uses the meta cache, and labels follow that device as `long`.
- Same-device alignment keeps autograd: `sum().backward()` reaches the CPU input and the CPU weight.
- One `_fine_tuning` step rebuilds a CPU network and ignores a meta cache. Weights stay finite on CPU.

**CUDA** `@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")` — collected; skipped on this CPU-only VM (same gap as #164):

- `parent._apply(cuda)` and `parent.to("cuda")` do not call `.to`. CPU features score on CUDA and match a CPU weight copy.
- Forcing the DBN cache back to CPU leaves CUDA parameters in charge. The CPU caller keeps `data_ptr`.
- A batch that is already on CUDA keeps feature and label storage. Logits are finite on CUDA.
- CUDA → CPU → CUDA is reread. `.to` is not called.
- float64 survives a CUDA `_apply`. `parent.to(dtype=torch.float64)` after a CUDA `_apply` stays on CUDA, does not call the child `.to`, and keeps float64 logits. Caller labels stay `long` on CPU.
- Backward from a CPU batch reaches CUDA weights. The input gradient stays on CPU. Caller `data_ptr` stays put.
- A CUDA batch is copied onto CPU parameters. Caller storage stays on CUDA.
- One training step sends CPU `int32` indices to CUDA as `long`. The caller stays `int32` on CPU. The weight gradient is finite and nonzero on CUDA. A CPU cache does not pull the step back.

**Not in scope:** `src/kaiwu/torch_plugin/dbn.py` (#242 / #1349). RBM `get_hidden` (#1334). `get_layer_activations` still uses the cached device (#1367). Pretrained weight init (#1353). Classifier-mode predict (#1369). `dbn_trainer.py`. Copying parameter *values* out of `meta` with `.to()` (that raises; the return trip uses `_apply`). Materializing the training scalar on meta: `loss.item()` raises `RuntimeError` after the device check, which is the signal the meta training tests assert.

## Dedupe

| Item | Relationship |
|------|----------------|
| **Merged #164** | Review bar: `tests/test_bm_device_sync.py` (parent move does not call child `.to`, `meta` on every machine, CUDA class behind `@skipUnless`, dtype-only move keeps the device). This PR follows that split for fine-tune batches. The live location is the linear parameter, reread every call. |
| **#242 / #1349** (open) | `src/kaiwu/torch_plugin/dbn.py`. This PR does not edit that file. |
| **#1334** (open) | RBM `get_hidden` / `get_visible`. Different file. |
| **#1367** (open) | Pins `get_layer_activations`, which still uses the cached device. Not this path. |
| **#1353** (open) | Pretrained weight init. Unchanged. |
| **#1369** (open) | Classifier-mode `predict`. Different path. |
| **#1336 / #1338 / #1340 / #1359** (open) | Pin activation map, optimizer/loader factories, network structure, and parameter dump. Those methods are unchanged. |
| **#1376** (open) | ESM mask / token-id device. Same #164 test shape, different file. Its embedding-index gap is the class-index check here. |
| **#1268 / #1275 / #1290–#1292 / #1343–#1348 / #1370–#1375 / #1377** (open) | Other modules. |
| **#877** (open) | `dbn_trainer.py` logging. Different file. |

No other open PR edits `example/dbn_digits/supervised_dbn_digits.py` for this fine-tune batch path.

## Validation

CPU-only VM, CPython 3.12.3, `torch==2.7.0+cpu`. CUDA tests were collected and skipped.

| Command | Result |
|---------|--------|
| `python3 -m unittest tests.test_dbn_finetune_parameter_device -v` | **18 passed, 10 skipped** (CUDA) |
| `python3 -m black --check` on the example and the test | clean |
| `PYTHONPATH=example/dbn_digits:src python3 -m pylint tests/test_dbn_finetune_parameter_device.py` | **10.00/10** |
| `python3 -m pylint example/dbn_digits/supervised_dbn_digits.py` | **7.83/10**, same score as the parent revision of this file. Pre-existing example noise was not cleaned. |

```bash
python3 -m unittest tests.test_dbn_finetune_parameter_device -v
python3 -m black --check example/dbn_digits/supervised_dbn_digits.py tests/test_dbn_finetune_parameter_device.py
PYTHONPATH=example/dbn_digits:src python3 -m pylint tests/test_dbn_finetune_parameter_device.py
python3 -m pylint example/dbn_digits/supervised_dbn_digits.py
```

## Commits (fork)

| SHA | Summary |
|-----|---------|
| `71a2a9ddec8d7238daa1cb773f067b11376d8e34` | place fine-tune batches on the live parameter device |
| branch head | lock class-index device, `parent.to(meta)`, and the zero-storage return trip |

## Merge-ready checklist

- [x] Fork branch `cursor/dbn-finetune-parameter-device-2bd3` pushed to TNHTH
- [x] Harden pass matches the #164 bar (CPU / `meta` executed; CUDA class present and skipped without a GPU)
- [ ] Maintainer review
