"""get_hidden(requires_grad=True) must keep the autograd graph."""

import torch

from kaiwu.torch_plugin import RestrictedBoltzmannMachine


def test_get_hidden_requires_grad_keeps_the_graph() -> None:
    torch.manual_seed(0)
    model = RestrictedBoltzmannMachine(num_visible=3, num_hidden=2)
    inputs = torch.rand(3, 3, requires_grad=True)

    output = model.get_hidden(inputs, requires_grad=True)
    output.sum().backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_get_hidden_without_requires_grad_detaches() -> None:
    model = RestrictedBoltzmannMachine(num_visible=3, num_hidden=2)
    inputs = torch.rand(3, 3, requires_grad=True)

    output = model.get_hidden(inputs)
    assert not output.requires_grad
