"""Upper-tail accuracy of QVAE truncated-exponential inverse-CDF sampling."""

import math

import pytest
import torch

from kaiwu.torch_plugin.qvae_dist_util import Exponential, MixtureGeneric


def _uniform_draws(dtype=torch.float32):
    """Include the closest float32 draw below one without using an invalid endpoint."""
    return torch.tensor(
        [0.0, 0.1, 0.5, 0.9, 0.99, 1.0 - 2**-10, 1.0 - 2**-20, 1.0 - 2**-24],
        dtype=dtype,
    ).reshape(2, 4)


def _reference_quantiles(beta, probabilities):
    """Compute double-precision quantiles while retaining each survival component."""
    return torch.tensor(
        [
            -math.log((1.0 - value) + value * math.exp(-beta)) / beta
            for value in probabilities.flatten().tolist()
        ],
        dtype=probabilities.dtype,
    ).reshape_as(probabilities)


@pytest.mark.parametrize("beta", [1.0, 2.0, 10.0, 15.0, 18.0, 20.0, 80.0, 200.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_upper_tail_samples_match_quantiles_and_probability_transform(beta, dtype, monkeypatch):
    """Retain the exp(-beta) tail mass when uniform probabilities approach one."""
    probabilities = _uniform_draws(dtype)
    monkeypatch.setattr(torch, "rand", lambda shape: probabilities.reshape(shape).clone())
    distribution = Exponential(beta)

    samples = distribution.sample(tuple(probabilities.shape))

    assert samples.shape == probabilities.shape
    assert samples.dtype == dtype
    assert torch.isfinite(samples).all()
    assert torch.all((samples >= 0.0) & (samples <= 1.0))
    assert torch.all(samples.flatten()[1:] > samples.flatten()[:-1])
    torch.testing.assert_close(
        samples, _reference_quantiles(beta, probabilities), rtol=2e-6, atol=2e-7
    )
    # Check the inverse-CDF probability transform in double precision, including
    # relative survival accuracy where a rounded CDF of one would hide errors.
    rate = float(beta)
    survival = torch.exp(-rate * samples.double())
    expected_survival = (1.0 - probabilities.double()) + math.exp(-rate) * probabilities.double()
    torch.testing.assert_close(survival, expected_survival, rtol=3e-6, atol=0.0)
    cdf = -torch.expm1(-rate * samples.double()) / -math.expm1(-rate)
    torch.testing.assert_close(cdf, probabilities.double(), rtol=0.0, atol=2e-7)


@pytest.mark.parametrize("beta", [0.25, 0.5, 0.999])
def test_rates_below_one_keep_correct_quantiles(beta, monkeypatch):
    """Ordinary rates below one retain their existing valid sampling distribution."""
    probabilities = _uniform_draws()
    monkeypatch.setattr(torch, "rand", lambda shape: probabilities.reshape(shape).clone())
    distribution = Exponential(beta)
    samples = distribution.sample(tuple(probabilities.shape))
    assert torch.isfinite(samples).all()
    assert torch.all((samples >= 0.0) & (samples <= 1.0))
    torch.testing.assert_close(samples, _reference_quantiles(distribution.beta.item(), probabilities))


def test_mixed_tensor_rates_keep_broadcasting_and_quantile_accuracy(monkeypatch):
    """Select the tail formula per rate without requiring beta to be scalar."""
    probabilities = _uniform_draws()
    monkeypatch.setattr(torch, "rand", lambda shape: probabilities.reshape(shape).clone())
    distribution = Exponential([0.5, 10.0, 15.0, 18.0])
    rate = distribution.beta
    expected = torch.tensor(
        [
            -math.log((1.0 - value) + value * math.exp(-beta)) / beta
            for row in probabilities.tolist()
            for beta, value in zip(rate.tolist(), row)
        ]
    ).reshape_as(probabilities)
    torch.testing.assert_close(distribution.sample((2, 4)), expected, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("beta", [10.0, 15.0, 18.0])
def test_mixture_preserves_tail_samples_and_finite_gradients(beta, monkeypatch):
    """QVAE's mixture caller receives accurate bounded samples and finite gradients."""
    probabilities = _uniform_draws()
    monkeypatch.setattr(torch, "rand", lambda shape: probabilities.reshape(shape).clone())
    monkeypatch.setattr(torch, "rand_like", lambda value, **_kwargs: torch.full_like(value, 0.75))
    logits = torch.zeros_like(probabilities, requires_grad=True)
    distribution = MixtureGeneric(logits, smoothing_dist_beta=beta)

    samples = distribution.reparameterize(is_training=True)
    gradient = torch.autograd.grad(samples.sum(), logits)[0]

    torch.testing.assert_close(samples, _reference_quantiles(beta, probabilities), rtol=2e-6, atol=2e-7)
    assert torch.all((samples >= 0.0) & (samples <= 1.0))
    assert torch.isfinite(gradient).all()
