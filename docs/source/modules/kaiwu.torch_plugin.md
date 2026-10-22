# 模块手册

## Module contents

```{eval-rst}
.. automodule:: kaiwu.torch_plugin
   :members:
   :undoc-members:
   :show-inheritance:
```

## Full BM Gibbs random streams

`BoltzmannMachine.gibbs_sample` accepts a keyword-only `generator` that controls
initialization, random scan order and Bernoulli draws. It must be on the sampling
device. With `generator=None`, the existing global RNG path is preserved.

```python
import torch
from kaiwu.torch_plugin import BoltzmannMachine

model = BoltzmannMachine(num_nodes=8, device="cpu")
stream = torch.Generator(device="cpu").manual_seed(7)
samples = model.gibbs_sample(num_steps=100, num_sample=32, generator=stream)
next_draw_state = stream.get_state()
# Restore the random stream when reproducing the next sampling call:
resumed_stream = torch.Generator(device="cpu")
resumed_stream.set_state(next_draw_state)
```

A private stream is unaffected by other modules' global random draws and does
not consume the global stream. Save its state separately from model weights.
This restores random draws, not persistent Gibbs chain states; each call keeps
its existing initialization behavior. Reproducibility requires the same model,
device and sampling settings, and does not establish mixing or convergence.
Type and device mismatches are rejected before sampling begins. CPU device
aliases are accepted; CUDA streams must also match the resolved device index.
