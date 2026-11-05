"""Large smoothing beta must not underflow the mixture log-likelihood."""

import math

import pytest
import torch

from kaiwu.torch_plugin.qvae_dist_util import MixtureGeneric


def test_large_beta_log_prob_stays_finite():
    beta = 1000.0
    posterior = MixtureGeneric(torch.zeros(1, 1), beta)
    zeta = torch.full((1, 1), 0.5)

    log_prob = posterior.log_prob_per_var(zeta)

    expected = math.log(beta) - beta * 0.5 - math.log(1.0 - math.exp(-beta))
    assert torch.isfinite(log_prob).all()
    torch.testing.assert_close(log_prob, torch.full_like(zeta, expected), rtol=1e-5, atol=1e-4)


def test_large_beta_handles_saturated_logits():
    logits = torch.tensor([[-1000.0, 1000.0]])
    posterior = MixtureGeneric(logits, 1000.0)

    log_prob = posterior.log_prob_per_var(torch.full_like(logits, 0.5))

    assert torch.isfinite(log_prob).all()


def test_ordinary_beta_matches_the_direct_density():
    torch.manual_seed(0)
    logits = torch.randn(2, 3)
    samples = torch.rand(2, 3)
    posterior = MixtureGeneric(logits, 1.0)
    q = torch.sigmoid(logits)
    reference = torch.log(
        q * posterior.smoothing_dist.pdf(1.0 - samples)
        + (1.0 - q) * posterior.smoothing_dist.pdf(samples)
    )

    torch.testing.assert_close(
        posterior.log_prob_per_var(samples), reference, rtol=1e-5, atol=1e-6
    )
