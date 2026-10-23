"""Constructor contract for the feature-selection example models."""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "feature_selection"))

from feature_selection_models import SimpleLSTM, SimpleRNN, TinyCNN  # noqa: E402


@pytest.mark.parametrize("image_size", [0, -4, True, 2.5])
def test_tiny_cnn_rejects_a_non_positive_integer_image_size(image_size):
    """A bad image size fails at construction with a clear message."""
    with pytest.raises(ValueError, match="image_size"):
        TinyCNN(image_size)


@pytest.mark.parametrize("feature_dim", [0, -3, True, 1.5])
def test_recurrent_models_reject_a_non_positive_integer_feature_dim(feature_dim):
    """Both recurrent classifiers validate their input feature count."""
    with pytest.raises(ValueError, match="feature_dim"):
        SimpleRNN(feature_dim)
    with pytest.raises(ValueError, match="feature_dim"):
        SimpleLSTM(feature_dim)


def test_models_still_accept_valid_dimensions():
    """Valid positive integers keep producing two logits per sample."""
    images = torch.randn(2, 4 * 4)
    assert TinyCNN(4)(images).shape == (2, 2)

    sequence = torch.randn(2, 5, 3)
    assert SimpleRNN(3)(sequence).shape == (2, 2)
    assert SimpleLSTM(3)(sequence).shape == (2, 2)
