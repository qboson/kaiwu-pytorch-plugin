"""BoltzmannMachine.forward must match the quadratic energy formula."""

import torch

from kaiwu.torch_plugin import BoltzmannMachine


def _model():
    quadratic = torch.tensor(
        [[0.5, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 2.0, 0.3]], dtype=torch.float32
    )
    bias = torch.tensor([0.2, -0.4, 0.1], dtype=torch.float32)
    return BoltzmannMachine(3, quadratic_coef=quadratic, linear_bias=bias)


def test_energy_matches_the_quadratic_formula():
    model = _model()
    states = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
    )
    weight = model.symmetrized_quadratic_coef().detach()

    energy = model(states)

    expected = -states @ model.linear_bias.detach() - 0.5 * torch.sum(
        (states @ weight) * states, dim=-1
    )
    torch.testing.assert_close(energy, expected)
