"""Contract test for the Exponential smoothing-distribution beta."""

from __future__ import annotations

import math

import pytest
import torch

from kaiwu.torch_plugin.qvae_dist_util import Exponential


@pytest.mark.parametrize("beta", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_a_non_positive_or_non_finite_beta(beta):
    with pytest.raises(ValueError, match="beta"):
        Exponential(beta)


def test_accepts_a_positive_beta_and_keeps_float32():
    distribution = Exponential(2.0)

    assert distribution.beta.dtype == torch.float32
    assert float(distribution.beta) == 2.0
    assert math.isfinite(float(distribution.beta))
