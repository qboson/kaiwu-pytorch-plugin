"""Contract test for the ``Exponential`` smoothing distribution."""

from __future__ import annotations

import math

import pytest
import torch

from kaiwu.torch_plugin.qvae_dist_util import Exponential


def test_pdf_is_normalized_on_the_unit_interval():
    distribution = Exponential(2.0)
    grid = torch.linspace(0.0, 1.0, 20001)

    integral = torch.trapz(distribution.pdf(grid), grid)

    assert float(integral) == pytest.approx(1.0, abs=5e-3)
    assert torch.all(distribution.pdf(grid) >= 0)


def test_cdf_endpoints_cover_the_unit_interval():
    distribution = Exponential(1.5)

    assert float(distribution.cdf(torch.tensor(0.0))) == pytest.approx(0.0, abs=1e-6)
    assert float(distribution.cdf(torch.tensor(1.0))) == pytest.approx(1.0, abs=1e-6)
    mid = torch.tensor(0.4)
    assert 0.0 < float(distribution.cdf(mid)) < 1.0


def test_pdf_matches_the_truncated_exponential_closed_form():
    beta = 0.8
    distribution = Exponential(beta)
    grid = torch.linspace(0.0, 1.0, 7)
    denominator = 1.0 - math.exp(-beta)

    expected = beta * torch.exp(-beta * grid) / denominator
    assert torch.allclose(distribution.pdf(grid), expected, atol=1e-6)
