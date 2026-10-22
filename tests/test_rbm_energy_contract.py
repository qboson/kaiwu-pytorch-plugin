"""Pin the Restricted Boltzmann Machine energy formula."""

import torch

from kaiwu.torch_plugin import RestrictedBoltzmannMachine


def test_energy_matches_the_definition() -> None:
    torch.manual_seed(0)
    model = RestrictedBoltzmannMachine(num_visible=3, num_hidden=2)
    with torch.no_grad():
        model.linear_bias.copy_(torch.randn_like(model.linear_bias))
        model.quadratic_coef.copy_(torch.randn_like(model.quadratic_coef))
    states = torch.bernoulli(torch.full((5, 5), 0.5))

    visible = states[:, :3]
    hidden = states[:, 3:]
    expected = -states @ model.linear_bias - torch.sum(
        visible.matmul(model.quadratic_coef) * hidden, dim=-1
    )

    torch.testing.assert_close(model(states), expected)


def test_energy_is_additive_over_independent_samples() -> None:
    torch.manual_seed(1)
    model = RestrictedBoltzmannMachine(num_visible=2, num_hidden=2)
    with torch.no_grad():
        model.linear_bias.copy_(torch.randn_like(model.linear_bias))
        model.quadratic_coef.copy_(torch.randn_like(model.quadratic_coef))
    states = torch.bernoulli(torch.full((4, 4), 0.5))

    batched = model(states)
    singles = torch.cat([model(state.unsqueeze(0)) for state in states])
    torch.testing.assert_close(batched, singles)
