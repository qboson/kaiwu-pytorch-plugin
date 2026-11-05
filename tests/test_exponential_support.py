"""Exponential smoothing distribution must honour its [0, 1] support."""

import math

import pytest
import torch

from kaiwu.torch_plugin.qvae_dist_util import Exponential


def _dist() -> Exponential:
    return Exponential(1.0)


@pytest.mark.parametrize("zeta", [-0.5, 1.5])
def test_pdf_is_zero_outside_the_support(zeta) -> None:
    assert _dist().pdf(torch.tensor([zeta])).item() == 0.0


@pytest.mark.parametrize("zeta, expected", [(-0.5, 0.0), (1.5, 1.0)])
def test_cdf_is_saturated_outside_the_support(zeta, expected) -> None:
    assert _dist().cdf(torch.tensor([zeta])).item() == expected


@pytest.mark.parametrize("zeta", [-0.5, 1.5])
def test_log_pdf_is_negative_infinity_outside_the_support(zeta) -> None:
    assert _dist().log_pdf(torch.tensor([zeta])).item() == -math.inf


def test_inside_the_support_the_formulas_are_unchanged() -> None:
    dist = _dist()
    zeta = torch.tensor([0.25, 0.75])
    beta = 1.0
    norm = 1 - math.exp(-beta)
    assert torch.allclose(
        dist.pdf(zeta), beta * torch.exp(-beta * zeta) / norm
    )
    assert torch.allclose(dist.cdf(zeta), (1 - torch.exp(-beta * zeta)) / norm)
    assert torch.allclose(dist.cdf(torch.tensor([0.0])), torch.zeros(1))
    assert torch.allclose(dist.cdf(torch.tensor([1.0])), torch.ones(1))
