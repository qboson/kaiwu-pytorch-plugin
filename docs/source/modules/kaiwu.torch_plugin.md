# 模块手册

## Module contents

```{eval-rst}
.. automodule:: kaiwu.torch_plugin
   :members:
   :undoc-members:
   :show-inheritance:
```

## Feature-selection checkpoints

`FeatureSelectionWrapper.state_dict()` saves the model weights, feature mask,
and completed training epochs. The epoch count is an int64 scalar tensor under
`_extra_state` (with the usual module prefix when nested). Restoring it preserves
the `mask_update_epochs` schedule when continuing with `fit_weights`.

Rebuild the wrapper with the same constructor configuration, including solver
options and mask-update interval, before loading its state. Save and restore the
optimizer's `state_dict()` separately for continued weight training. These tensor
states support `torch.save` and `torch.load(..., weights_only=True)`.

Older checkpoints containing only weights and mask still load with `strict=True`;
their unknown epoch count starts at zero. Unrelated missing or unexpected keys
remain subject to the normal strict-loading checks.
