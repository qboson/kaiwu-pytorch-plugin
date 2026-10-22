"""The BM objective must be the mean-energy difference."""

import torch

from kaiwu.torch_plugin import BoltzmannMachine


def _model():
    quadratic = torch.tensor(
        [[0.5, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 2.0, 0.3]], dtype=torch.float32
    )
    bias = torch.tensor([0.2, -0.4, 0.1], dtype=torch.float32)
    return BoltzmannMachine(3, quadratic_coef=quadratic, linear_bias=bias)


def test_objective_is_the_mean_energy_difference():
    model = _model()
    positive = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    negative = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    objective = model.objective(positive, negative)

    expected = model(positive).mean() - model(negative).mean()
    torch.testing.assert_close(objective, expected)
