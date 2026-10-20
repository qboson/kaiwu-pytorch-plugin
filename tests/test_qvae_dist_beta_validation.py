# -*- coding: utf-8 -*-
"""The exponential smoothing distribution requires a positive finite beta."""

from __future__ import annotations

import math

import pytest
import torch

from kaiwu.torch_plugin.qvae_dist_util import Exponential, MixtureGeneric


@pytest.mark.parametrize(
    "beta",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf"), [0.5, 1.0]],
)
def test_invalid_beta_is_rejected(beta) -> None:
    """Zero, negative, non-finite, or non-scalar beta must fail fast."""
    with pytest.raises(ValueError, match="finite positive scalar"):
        Exponential(beta)


def test_small_positive_beta_remains_supported() -> None:
    """Small-but-positive beta stays valid and produces finite unit-interval draws."""
    distribution = Exponential(1e-8)

    samples = distribution.sample((256,))

    assert torch.isfinite(samples).all()
    assert ((samples >= 0) & (samples < 1)).all()


def test_mixture_rejects_invalid_beta_through_smoothing_distribution() -> None:
    """MixtureGeneric delegates validation to its smoothing distribution."""
    with pytest.raises(ValueError, match="finite positive scalar"):
        MixtureGeneric(torch.zeros(2), 0.0)


def test_positive_beta_density_is_finite() -> None:
    """A regular beta still has a finite density and log-density."""
    distribution = Exponential(2.0)
    zeta = torch.tensor([0.0, 0.5, 1.0])

    assert torch.isfinite(distribution.pdf(zeta)).all()
    assert torch.isfinite(distribution.log_pdf(zeta)).all()
    assert math.isfinite(float(distribution.cdf(torch.tensor(0.5))))
