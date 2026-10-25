"""Contract test for ``sigmoid_cross_entropy_with_logits``."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from kaiwu.torch_plugin.qvae_dist_util import sigmoid_cross_entropy_with_logits


def test_matches_the_reference_binary_cross_entropy():
    logits = torch.tensor([[-2.0, 0.0, 3.0], [1.0, -1.0, 0.5]])
    labels = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])

    expected = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

    assert torch.allclose(sigmoid_cross_entropy_with_logits(logits, labels), expected)


def test_stays_finite_for_extreme_logits():
    logits = torch.tensor([1000.0, -1000.0, 60.0])
    labels = torch.tensor([1.0, 0.0, 1.0])

    loss = sigmoid_cross_entropy_with_logits(logits, labels)

    assert torch.isfinite(loss).all()
    assert torch.allclose(loss, torch.zeros(3), atol=1e-5)


def test_preserves_the_input_shape():
    logits = torch.randn(2, 3, 4)
    labels = torch.randint(0, 2, (2, 3, 4)).float()

    assert sigmoid_cross_entropy_with_logits(logits, labels).shape == (2, 3, 4)
