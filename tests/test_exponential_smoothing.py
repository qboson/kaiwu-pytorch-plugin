"""Numerical behavior of the truncated exponential used by QVAE smoothing."""

import math

import pytest
import torch

from kaiwu.torch_plugin.qvae_dist_util import Exponential, MixtureGeneric


@pytest.mark.parametrize("beta", [1e-8, 1e-5, 0.01, 1.0, 10.0, 80.0, 200.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_density_and_cdf_match_double_precision_reference(beta, dtype):
    """Small positive rates must not lose their normalization to cancellation."""
    distribution = Exponential(beta)
    rate = distribution.beta.item()
    normalizer = -math.expm1(-rate)
    inputs = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=dtype)
    expected_pdf = torch.tensor(
        [rate * math.exp(-rate * value) / normalizer for value in inputs.tolist()],
        dtype=dtype,
    )
    expected_cdf = torch.tensor(
        [-math.expm1(-rate * value) / normalizer for value in inputs.tolist()],
        dtype=dtype,
    )
    expected_log_pdf = torch.tensor(
        [math.log(rate / normalizer) - rate * value for value in inputs.tolist()],
        dtype=dtype,
    )

    for actual, expected in (
        (distribution.pdf(inputs), expected_pdf),
        (distribution.cdf(inputs), expected_cdf),
        (distribution.log_pdf(inputs), expected_log_pdf),
    ):
        assert actual.dtype == dtype
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("beta", [1e-8, 1e-5, 1.0, 10.0, 200.0])
def test_cdf_has_correct_boundaries_and_density_derivative(beta):
    """The CDF remains monotone and its input derivative equals the PDF."""
    distribution = Exponential(beta)
    inputs = torch.linspace(0.0, 1.0, 1025, requires_grad=True)
    cdf = distribution.cdf(inputs)

    assert cdf[0].item() == 0.0
    assert cdf[-1].item() == 1.0
    assert torch.isfinite(cdf).all()
    assert torch.all(cdf[1:] >= cdf[:-1])
    density_gradient = torch.autograd.grad(cdf.sum(), inputs)[0]
    torch.testing.assert_close(
        density_gradient, distribution.pdf(inputs), rtol=2e-6, atol=2e-6
    )
    log_gradient = torch.autograd.grad(distribution.log_pdf(inputs).sum(), inputs)[0]
    torch.testing.assert_close(log_gradient, torch.full_like(inputs, -beta))


@pytest.mark.parametrize("beta", [1e-8, 1e-5, 0.01, 1.0, 10.0, 80.0, 200.0])
def test_samples_match_inverse_cdf_quantiles_and_probability_transform(beta, monkeypatch):
    """Fixed uniform draws provide a deterministic check of sampling accuracy."""
    probabilities = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0 - 2**-24])
    monkeypatch.setattr(torch, "rand", lambda shape: probabilities.reshape(shape).clone())
    distribution = Exponential(beta)
    rate = distribution.beta.item()
    expected = torch.tensor(
        [
            -math.log1p(math.expm1(-rate) * value) / rate
            for value in probabilities.tolist()
        ]
    )

    samples = distribution.sample((len(probabilities),))

    assert torch.isfinite(samples).all()
    assert torch.all((samples >= 0.0) & (samples <= 1.0))
    assert torch.all(samples[1:] > samples[:-1])
    torch.testing.assert_close(samples, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        distribution.cdf(samples), probabilities, rtol=2e-6, atol=2e-6
    )


def test_small_positive_rate_approaches_uniform_distribution(monkeypatch):
    """A nearly flat smoothing distribution must not collapse all draws to zero."""
    probabilities = torch.linspace(0.0, 0.99, 100)
    monkeypatch.setattr(torch, "rand", lambda shape: probabilities.reshape(shape).clone())
    distribution = Exponential(1e-8)

    torch.testing.assert_close(distribution.pdf(probabilities), torch.ones_like(probabilities))
    torch.testing.assert_close(distribution.cdf(probabilities), probabilities)
    torch.testing.assert_close(distribution.log_pdf(probabilities), torch.zeros_like(probabilities))
    torch.testing.assert_close(distribution.sample((100,)), probabilities)


@pytest.mark.parametrize("beta", [1e-8, 1e-5, 1.0, 10.0])
def test_mixture_smoothing_remains_finite_with_small_positive_rate(beta):
    """The QVAE caller must receive finite samples, log probabilities, and gradients."""
    logits = torch.tensor([[-4.0, 0.0, 4.0]] * 8, requires_grad=True)
    distribution = MixtureGeneric(logits, smoothing_dist_beta=beta)

    samples = distribution.reparameterize(is_training=True)
    log_probabilities = distribution.log_prob_per_var(samples)
    log_ratios = distribution.log_ratio(samples)
    gradient = torch.autograd.grad(
        (samples + log_probabilities + log_ratios).sum(), logits
    )[0]

    assert samples.shape == logits.shape
    assert torch.all((samples >= 0.0) & (samples <= 1.0))
    for value in (samples, log_probabilities, log_ratios, gradient):
        assert torch.isfinite(value).all()


def test_zero_rate_keeps_existing_undefined_results():
    """This stability fix does not introduce a new beta=0 distribution contract."""
    distribution = Exponential(0.0)
    inputs = torch.tensor([0.0, 0.5, 1.0])

    for value in (
        distribution.pdf(inputs),
        distribution.cdf(inputs),
        distribution.log_pdf(inputs),
        distribution.sample((3,)),
    ):
        assert torch.isnan(value).all()


def test_negative_rate_keeps_existing_log_density_behavior():
    """Negative rates retain the existing log-density behavior without new validation."""
    distribution = Exponential(-1.0)
    assert torch.isnan(distribution.log_pdf(torch.tensor([0.0, 0.5, 1.0]))).all()
