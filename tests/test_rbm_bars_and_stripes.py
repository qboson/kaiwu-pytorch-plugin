"""Exercise sampler integration and exact visible metrics on a tiny RBM."""

import math

import numpy as np
import pytest
import torch

from example.rbm_bars_and_stripes import (
    EnumeratedBoltzmannSampler,
    bars_and_stripes,
    run_benchmark,
    visible_metrics,
)
from kaiwu.torch_plugin import RestrictedBoltzmannMachine


@pytest.mark.parametrize("side, expected", [(2, 6), (3, 14)])
def test_pattern_set_has_each_constant_image_only_once(side, expected):
    data = bars_and_stripes(side)
    assert tuple(data.shape) == (expected, side * side)
    assert (data.sum(dim=1) == 0).sum() == 1
    assert (data.sum(dim=1) == side * side).sum() == 1


def test_exact_metrics_equal_baseline_for_uniform_visible_distribution():
    model = RestrictedBoltzmannMachine(
        4, 2, quadratic_coef=torch.zeros(4, 2), device="cpu",
    )
    metrics = visible_metrics(model, bars_and_stripes(2))
    assert metrics["pattern_mass"] == pytest.approx(6 / 16)
    assert metrics["mean_nll"] == pytest.approx(math.log(16))


def test_sampler_obeys_the_boltzmann_bias_and_reference_spin():
    model = RestrictedBoltzmannMachine(
        1, 1, quadratic_coef=torch.zeros(1, 1), linear_bias=torch.tensor([1., 0.]),
        device="cpu",
    )
    sampler = EnumeratedBoltzmannSampler(2, reads=4000, seed=40)
    samples = model.sample(sampler)
    np.testing.assert_allclose(samples.mean(dim=0).numpy(), [1 / (1 + math.exp(-1)), 0.5], atol=0.025)


def test_training_improves_pattern_probability_at_the_documented_seed():
    result = run_benchmark(steps=300, seed=42)
    assert result["trained"]["pattern_mass"] > result["initial"]["pattern_mass"]
    assert result["trained"]["mean_nll"] < result["initial"]["mean_nll"]
