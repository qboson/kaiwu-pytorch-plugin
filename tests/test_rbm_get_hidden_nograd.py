"""get_hidden must detach gradients unless requires_grad is set."""

import torch

from kaiwu.torch_plugin import RestrictedBoltzmannMachine


def _model():
    torch.manual_seed(0)
    return RestrictedBoltzmannMachine(2, 3, device=torch.device("cpu"))


def test_default_propagation_detaches_gradients():
    model = _model()
    visible = torch.rand(2, 2, requires_grad=True)

    states = model.get_hidden(visible)

    assert states.requires_grad is False


def test_requires_grad_true_keeps_the_graph():
    model = _model()
    visible = torch.rand(2, 2, requires_grad=True)

    states = model.get_hidden(visible, requires_grad=True)

    assert states.requires_grad is True
