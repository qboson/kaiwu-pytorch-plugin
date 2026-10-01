# 模块手册

## Module contents

```{eval-rst}
.. automodule:: kaiwu.torch_plugin
   :members:
   :undoc-members:
   :show-inheritance:
```

## Mask learning with different input sample layouts

`FeatureSelectionWrapper` accepts keyword-only `input_batch_axis=0` to identify
independent input samples during `compute_mask_derivatives` and `update_mask`.
The same configuration applies to automatic updates in `fit_weights`.
`input_feature_axis` identifies selectable features; these two axes must be
different when collecting derivative batches. Neither axis is inferred from
dimension sizes. Targets always use a leading sample dimension `(batch, ...)`.

| Input layout | `input_feature_axis` | `input_batch_axis` |
| --- | --- | --- |
| `(batch, features)` | `-1` | `0` (default) |
| `(features, batch)` | `0` | `1` or `-1` |
| `(batch, features, steps)` | `1` | `0` (default) |
| `(features, batch, steps)` | `0` | `1` or `-2` |

For example, wrap a model that accepts `(features, batch)` inputs and returns
batch-first predictions with `FeatureSelectionWrapper(model, feature_dim=3,
input_feature_axis=0, input_batch_axis=1)`. Each loader item supplies those inputs
and batch-first targets with the same sample count.

Input batches are counted, concatenated, and truncated along the configured
sample axis. `max_samples` selects the first observations yielded by the loader,
including a partial final batch; it does not truncate features or time steps.
Different batches can have different sample counts. Negative sample axes count
backwards from the input tensor's last dimension and must be in its dimension
range. Invalid sample axes, coincident sample/feature axes, scalar targets, and
unequal input/target sample counts raise `ValueError` during derivative collection.
These batch requirements do not restrict `apply_mask` or `forward` on a single
feature vector. Existing positional constructor arguments remain unchanged.
