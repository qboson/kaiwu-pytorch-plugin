"""Pin the Gaussian-Bernoulli RBM energy and marginal energy formulas."""

import torch

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


def _model() -> GaussianBernoulliRestrictedBoltzmannMachine:
    torch.manual_seed(0)
    model = GaussianBernoulliRestrictedBoltzmannMachine(num_visible=3, num_hidden=2)
    with torch.no_grad():
        model.mu.copy_(torch.randn_like(model.mu))
        model.log_var.copy_(torch.randn_like(model.log_var) * 0.1)
        model.quadratic_coef.copy_(torch.randn_like(model.quadratic_coef))
        model.linear_bias.copy_(torch.randn_like(model.linear_bias))
    return model


def test_energy_matches_the_definition() -> None:
    model = _model()
    states = torch.cat(
        [
            torch.randn(4, 3),
            torch.bernoulli(torch.full((4, 2), 0.5)),
        ],
        dim=1,
    )
    gaussian = states[:, :3]
    bernoulli = states[:, 3:]
    expected = (
        0.5 * torch.sum((gaussian - model.mu).square() / model.var, dim=-1)
        - torch.sum((gaussian / model.var) @ model.quadratic_coef * bernoulli, dim=-1)
        - bernoulli @ model.linear_bias
    )
    torch.testing.assert_close(model.energy(states), expected)


def test_marginal_energy_matches_the_definition() -> None:
    model = _model()
    gaussian = torch.randn(4, 3)
    expected = 0.5 * torch.sum(
        (gaussian - model.mu).square() / model.var, dim=-1
    ) - torch.sum(
        torch.nn.functional.softplus(
            (gaussian / model.var) @ model.quadratic_coef + model.linear_bias
        ),
        dim=-1,
    )
    torch.testing.assert_close(model.marginal_energy(gaussian), expected)
