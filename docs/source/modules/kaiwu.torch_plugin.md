# 模块手册

## Module contents

```{eval-rst}
.. automodule:: kaiwu.torch_plugin
   :members:
   :undoc-members:
   :show-inheritance:
```

## DBN inference precision

For built and trained `UnsupervisedDBN` models, NumPy inputs to `forward` and
`transform` are converted to each RBM's current parameter dtype and device.
`reconstruct` and `reconstruct_with_rbm` align inputs and hidden states with the
selected RBM's weights before computing reconstructions and per-sample errors.
This supports already-built models converted with PyTorch `double`, `half` or
`bfloat16`, including stacks whose layers use different dtypes.

All four methods keep their NumPy outputs. Hidden features retain the dtype and
precision supplied by the RBM's `get_hidden` implementation. An RBM returning
float32 states still limits hidden-state precision to float32. Reconstruction
outputs follow the selected parameter dtype. NumPy cannot represent BF16, so
only BF16 results are exported as float32. This conversion does not alter model
parameters. An empty trained stack retains its original float32 NumPy output.
