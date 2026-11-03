"""The truncated exponential must stay finite when beta is tiny."""

import math

import pytest
import torch

from kaiwu.torch_plugin.qvae_dist_util import Exponential, MixtureGeneric


ZETA = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])


@pytest.mark.parametrize("beta", [1e-8, 1e-7, 1e-6])
def test_small_beta_density_stays_finite(beta):
    """beta -> 0 tends to a uniform density, not inf/NaN from cancellation."""
    distribution = Exponential(beta)

    pdf = distribution.pdf(ZETA)
    cdf = distribution.cdf(ZETA)
    log_pdf = distribution.log_pdf(ZETA)

    assert torch.isfinite(pdf).all()
    assert torch.isfinite(cdf).all()
    assert torch.isfinite(log_pdf).all()
    torch.testing.assert_close(cdf, ZETA, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(pdf, torch.ones_like(ZETA), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(log_pdf, torch.zeros_like(ZETA), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("beta", [0.5, 1.0, 5.0])
def test_moderate_beta_matches_the_closed_form(beta):
    """The expm1 rewrite must not change the ordinary density."""
    zeta = torch.linspace(0.0, 1.0, 11)
    denominator = 1.0 - math.exp(-beta)
    reference_pdf = beta * torch.exp(-beta * zeta) / denominator
    reference_cdf = (1.0 - torch.exp(-beta * zeta)) / denominator
    distribution = Exponential(beta)

    torch.testing.assert_close(distribution.pdf(zeta), reference_pdf, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(distribution.cdf(zeta), reference_cdf, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        distribution.log_pdf(zeta), torch.log(reference_pdf), rtol=1e-5, atol=1e-6
    )


def test_small_beta_samples_span_the_unit_interval(monkeypatch):
    """The inverse CDF must not collapse every draw to zero for tiny beta."""
    draws = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0 - 2**-24])
    monkeypatch.setattr(torch, "rand", lambda shape: draws.reshape(shape).clone())

    samples = Exponential(1e-8).sample((5,))

    assert torch.isfinite(samples).all()
    torch.testing.assert_close(samples, draws, atol=1e-5, rtol=1e-5)


def test_large_beta_mixture_log_prob_does_not_underflow():
    """Multiplying tiny densities must not turn a finite log-likelihood into -inf."""
    beta = 1000.0
    posterior = MixtureGeneric(torch.zeros(1, 1), beta)
    zeta = torch.full((1, 1), 0.5)

    log_prob = posterior.log_prob_per_var(zeta)

    expected = math.log(beta) - beta * 0.5 - math.log(1.0 - math.exp(-beta))
    assert torch.isfinite(log_prob).all()
    torch.testing.assert_close(log_prob, torch.full_like(zeta, expected), rtol=1e-5, atol=1e-4)


def test_large_beta_mixture_handles_saturated_logits():
    """Saturated q values must not produce -inf from log(0) in the mixture."""
    logits = torch.tensor([[-100.0, 100.0]])
    posterior = MixtureGeneric(logits, 1000.0)

    log_prob = posterior.log_prob_per_var(torch.full_like(logits, 0.5))

    assert torch.isfinite(log_prob).all()


def test_mixture_log_prob_matches_the_direct_density():
    """The log-space rewrite must agree with the original formula for ordinary beta."""
    torch.manual_seed(0)
    logits = torch.randn(2, 3)
    samples = torch.rand(2, 3)
    posterior = MixtureGeneric(logits, 1.0)
    q = torch.sigmoid(logits)
    reference = torch.log(
        q * posterior.smoothing_dist.pdf(1.0 - samples)
        + (1.0 - q) * posterior.smoothing_dist.pdf(samples)
    )

    torch.testing.assert_close(posterior.log_prob_per_var(samples), reference, rtol=1e-5, atol=1e-6)
