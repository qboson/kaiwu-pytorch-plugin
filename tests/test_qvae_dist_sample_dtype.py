"""Bernoulli utility sampling must follow the logits dtype."""

from __future__ import annotations

import torch

from kaiwu.torch_plugin.qvae_dist_util import FactorialBernoulliUtil


def test_reparameterize_preserves_float64_dtype() -> None:
    util = FactorialBernoulliUtil(torch.zeros(4, dtype=torch.float64))
    samples = util.reparameterize(is_training=False)
    assert samples.dtype == torch.float64


def test_reparameterize_preserves_float32_dtype() -> None:
    util = FactorialBernoulliUtil(torch.zeros(4, dtype=torch.float32))
    samples = util.reparameterize(is_training=False)
    assert samples.dtype == torch.float32


def test_reparameterize_returns_binary_values() -> None:
    util = FactorialBernoulliUtil(torch.randn(16, dtype=torch.float64))
    samples = util.reparameterize(is_training=False)
    assert set(samples.tolist()) <= {0.0, 1.0}
