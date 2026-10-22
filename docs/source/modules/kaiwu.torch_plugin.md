# 模块手册

## Module contents

```{eval-rst}
.. automodule:: kaiwu.torch_plugin
   :members:
   :undoc-members:
   :show-inheritance:
```

## Gaussian-Bernoulli Energy Evaluation and Gradients

Calling a Gaussian-Bernoulli RBM as a PyTorch module, `model(states)`, follows
the surrounding gradient context. Ordinary calls remain differentiable for
training, while calls inside `torch.no_grad()` or `torch.inference_mode()`
do not build backward graphs. As with other PyTorch modules, `model.eval()`
does not disable gradients by itself.

```python
import torch
from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine

model = GaussianBernoulliRestrictedBoltzmannMachine(2, 1, device="cpu")
states = torch.tensor([[0.2, -1.1, 1.0]])  # Gaussian nodes, then Bernoulli nodes.
energy = model(states)
energy.sum().backward()
assert model.mu.grad is not None

with torch.no_grad():
    evaluation_energy = model(states)
    assert not evaluation_energy.requires_grad
```

The separate `energy(states, enable_grad=False)` API keeps its explicit
gradient flag. Its default disables gradient recording, and
`energy(states, enable_grad=True)` enables it even inside `torch.no_grad()`.
An enclosing `torch.inference_mode()` still prevents autograd recording.
Use `model(states)` when gradient behavior should follow the caller's context.
