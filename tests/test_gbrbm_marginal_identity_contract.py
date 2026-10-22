"""Pin the Gaussian marginal energy identity of the GBRBM."""

import itertools

import torch

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


def _model() -> GaussianBernoulliRestrictedBoltzmannMachine:
    torch.manual_seed(0)
    model = GaussianBernoulliRestrictedBoltzmannMachine(num_visible=2, num_hidden=2)
    with torch.no_grad():
        model.mu.copy_(torch.randn_like(model.mu))
        model.log_var.copy_(torch.randn_like(model.log_var) * 0.1)
        model.quadratic_coef.copy_(torch.randn_like(model.quadratic_coef))
        model.linear_bias.copy_(torch.randn_like(model.linear_bias))
    return model


def test_marginal_energy_is_the_log_sum_exp_over_hidden_states() -> None:
    model = _model()
    gaussian = torch.randn(2, 2)

    marginal = model.marginal_energy(gaussian)
    energies = []
    for bits in itertools.product([0.0, 1.0], repeat=2):
        hidden = torch.tensor([bits])
        state = torch.cat([gaussian[:1], hidden], dim=1)
        energies.append(float(model.energy(state)[0]))
    expected = -torch.logsumexp(-torch.tensor(energies), dim=0)

    torch.testing.assert_close(marginal[0], expected, rtol=1e-5, atol=1e-6)
