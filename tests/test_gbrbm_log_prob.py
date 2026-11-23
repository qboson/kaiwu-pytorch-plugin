"""Exact Gaussian-Bernoulli densities match independently normalized mixtures."""

import math
from itertools import product

import pytest
import torch
from torch.distributions import Normal

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


@pytest.mark.parametrize("gaussian_visible", [True, False])
def test_gaussian_density_matches_normalized_mixture(gaussian_visible):
    dtype = torch.float64
    gaussian = torch.tensor([0.3, -0.5, 0.7], dtype=dtype)
    scale = torch.tensor([0.5, 1.5, 2.0], dtype=dtype)
    weights = torch.tensor([[0.2, -0.6], [0.7, 0.3], [-0.4, 0.8]], dtype=dtype)
    bias = torch.tensor([0.4, -0.1], dtype=dtype)
    dimensions = (3, 2) if gaussian_visible else (2, 3)
    model = GaussianBernoulliRestrictedBoltzmannMachine(
        *dimensions, is_visible_gaussian=gaussian_visible, dtype=dtype,
        quadratic_coef=weights, linear_bias=bias,
    )
    with torch.no_grad():
        model.mu.copy_(gaussian)
        model.log_var.copy_(2 * scale.log())

    points = torch.tensor([[0.1, -0.6, 0.3], [1.2, 1.3, -1.4]], dtype=dtype)
    hidden = torch.tensor(list(product((0.0, 1.0), repeat=2)), dtype=dtype)
    means = gaussian + hidden @ weights.T
    relative_mass = hidden @ bias + (
        ((means / scale).square() - (gaussian / scale).square()) / 2
    ).sum(dim=1)
    mixture_log_density = torch.logsumexp(
        relative_mass.log_softmax(dim=0)
        + Normal(means, scale).log_prob(points[:, None, :]).sum(dim=2),
        dim=1,
    )

    actual = model.gaussian_log_prob(points)
    torch.testing.assert_close(actual, mixture_log_density, rtol=1e-12, atol=1e-12)
    assert not actual.requires_grad


def test_log_partition_accumulates_more_than_one_enumeration_batch():
    bias = torch.linspace(-1.0, 1.0, steps=11, dtype=torch.float64)
    model = GaussianBernoulliRestrictedBoltzmannMachine(
        1, 11, dtype=torch.float64, linear_bias=bias, quadratic_coef=torch.zeros(1, 11)
    )
    expected = torch.logaddexp(torch.zeros_like(bias), bias).sum() + 0.5 * math.log(2 * math.pi)
    torch.testing.assert_close(model.log_partition(), expected, rtol=1e-12, atol=1e-12)


def test_exact_likelihood_rejects_models_beyond_enumeration_budget():
    model = GaussianBernoulliRestrictedBoltzmannMachine(2, 17)
    with pytest.raises(ValueError, match="at most 16 Bernoulli units"):
        model.gaussian_log_prob(torch.zeros(1, 2))
