"""init_parameter must set log_var and clip the parameters."""

import math

import torch

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


def test_init_parameter_sets_log_var_and_clips():
    torch.manual_seed(0)
    model = GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=3, num_hidden=2, dtype=torch.float64
    )

    with torch.no_grad():
        model.init_parameter(init_var=4.0, std=1.0, max_fold=1e-6)

    expected = torch.full_like(model.log_var, math.log(4.0))
    torch.testing.assert_close(model.log_var, expected)
    assert torch.all(model.mu.abs() <= 1e-6)
    assert torch.all(model.quadratic_coef.abs() <= 1e-6)
    assert torch.all(model.linear_bias.abs() <= 1e-6)
