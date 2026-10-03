# 模块手册

## Module contents

```{eval-rst}
.. automodule:: kaiwu.torch_plugin
   :members:
   :undoc-members:
   :show-inheritance:
```

## QVAE input layout

QVAE's existing flattening in `forward`, `energy`, and MSE reconstruction loss
accepts noncontiguous tensor layouts, including transposed image axes. Flattening
preserves logical element order and autograd; it uses a view when possible and
copies when strides require it. Input shape and reconstruction-loss conventions
are unchanged. Bernoulli image-shaped loss handling is a separate shape contract.
