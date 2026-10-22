"""Pin the enable_grad contract of GBRBM.energy."""

import torch

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


def _model() -> GaussianBernoulliRestrictedBoltzmannMachine:
    torch.manual_seed(0)
    model = GaussianBernoulliRestrictedBoltzmannMachine(num_visible=2, num_hidden=2)
    with torch.no_grad():
        model.quadratic_coef.copy_(torch.randn_like(model.quadratic_coef))
        model.linear_bias.copy_(torch.randn_like(model.linear_bias))
    return model


def test_energy_keeps_the_graph_when_requested() -> None:
    model = _model()
    states = torch.cat([torch.randn(2, 2), torch.zeros(2, 2)], dim=1)

    energy = model.energy(states, enable_grad=True)
    energy.sum().backward()

    assert model.quadratic_coef.grad is not None
    assert torch.isfinite(model.quadratic_coef.grad).all()


def test_energy_detaches_by_default() -> None:
    model = _model()
    states = torch.cat([torch.randn(1, 2), torch.zeros(1, 2)], dim=1)
    energy = model.energy(states)
    assert not energy.requires_grad
