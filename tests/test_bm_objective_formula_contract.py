"""Pin the Boltzmann machine objective formula."""

import torch

from kaiwu.torch_plugin import BoltzmannMachine


def test_objective_is_the_mean_energy_difference() -> None:
    torch.manual_seed(0)
    model = BoltzmannMachine(num_nodes=4)
    with torch.no_grad():
        model.quadratic_coef.copy_(torch.randn_like(model.quadratic_coef))
        model.linear_bias.copy_(torch.randn_like(model.linear_bias))
    positive = torch.bernoulli(torch.full((2, 4), 0.5))
    negative = torch.bernoulli(torch.full((3, 4), 0.5))

    objective = model.objective(positive, negative)

    expected = model(positive).mean() - model(negative).mean()
    torch.testing.assert_close(objective, expected)
