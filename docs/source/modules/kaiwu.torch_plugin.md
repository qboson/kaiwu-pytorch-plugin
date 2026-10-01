# 模块手册

## Module contents

```{eval-rst}
.. automodule:: kaiwu.torch_plugin
   :members:
   :undoc-members:
   :show-inheritance:
```

## QVAE Posterior Sample Precision

For floating-point logits, `FactorialBernoulliUtil.reparameterize(False)`
returns binary samples with the logits' dtype and device.
`MixtureGeneric.reparameterize` likewise returns latent samples with the
floating-point logits' dtype and device, so converting a QVAE with
PyTorch's `half()`, `bfloat16()` or `double()` keeps its encoder output,
posterior samples and decoder input compatible.

The existing float32 Exponential smoothing draws and component PDF/CDF
calculations are retained. The final mixture sample is converted to the
probability tensor's dtype after constructing its implicit-gradient expression, preserving
the gradient path to the encoder. Float32 and float64 sampling values,
random-number consumption and implicit-gradient behavior remain unchanged.
Integer and Boolean logits retain floating-point sampling outputs with the
computed sigmoid probability's dtype, avoiding truncation of mixture samples.
