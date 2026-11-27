"""GBRBM.init_parameter must validate its parameters."""

import math

import pytest
import torch

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


def _model():
    return GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=2, num_hidden=3, dtype=torch.float64
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"std": float("nan")},
        {"std": float("inf")},
        {"std": -1.0},
        {"init_var": 0.0},
        {"init_var": -1.0},
        {"init_var": float("nan")},
        {"init_var": float("inf")},
        {"max_fold": 0.0},
        {"max_fold": -2.0},
        {"max_fold": float("nan")},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _model().init_parameter(**kwargs)


def test_valid_parameters_initialize_finite_weights():
    model = _model()

    model.init_parameter(init_var=4.0, std=0.0, max_fold=16.0)

    assert torch.all(model.log_var == math.log(4.0))
    assert torch.count_nonzero(model.mu) == 0
    assert torch.count_nonzero(model.quadratic_coef) == 0
    assert torch.count_nonzero(model.linear_bias) == 0
