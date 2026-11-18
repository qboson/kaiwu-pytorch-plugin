"""Exponential smoothing is a normalized distribution on the unit interval."""
import math
import pytest
import torch
from kaiwu.torch_plugin.qvae_dist_util import Exponential, MixtureGeneric


@pytest.mark.parametrize("beta", [0.5, 2., 10.])
def test_density_and_cdf_respect_support(beta):
    dist = Exponential(beta)
    samples = torch.tensor([-math.inf, -1000., -0.2, 0., 0.5, 1., 1.2, math.inf])
    cdf = dist.cdf(samples)
    torch.testing.assert_close(cdf[:3], torch.zeros(3))
    torch.testing.assert_close(cdf[5:], torch.ones(3))
    expected = (1 - math.exp(-beta * 0.5)) / (1 - math.exp(-beta))
    assert cdf[4].item() == pytest.approx(expected, rel=1e-6)
    outside = torch.tensor([True, True, True, False, False, False, True, True])
    assert torch.equal(dist.pdf(samples)[outside], torch.zeros(5))
    assert torch.isneginf(dist.log_pdf(samples)[outside]).all()
    torch.testing.assert_close(dist.log_pdf(samples)[~outside].exp(), dist.pdf(samples)[~outside])


def test_outside_gradients_do_not_overflow():
    dist = Exponential(10.)
    samples = torch.tensor([-1000., -0.2, 0.5, 1.2], requires_grad=True)
    (dist.pdf(samples) + dist.cdf(samples)).sum().backward()
    assert torch.isfinite(samples.grad).all()
    torch.testing.assert_close(samples.grad[[0, 1, 3]], torch.zeros(3))


def test_mixture_cannot_assign_density_to_impossible_samples():
    mixture = MixtureGeneric(torch.tensor([0., 2.]), 2.)
    assert torch.isneginf(mixture.log_prob_per_var(torch.tensor([-0.1, 1.1]))).all()


def test_density_integrates_to_one_and_sampling_remains_bounded():
    dist = Exponential(2.)
    points = torch.linspace(0, 1, 10001)
    assert torch.trapezoid(dist.pdf(points), points).item() == pytest.approx(1., abs=2e-6)
    samples = dist.sample((1000,))
    assert ((samples >= 0) & (samples <= 1)).all()
