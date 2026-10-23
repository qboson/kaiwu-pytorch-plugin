"""Contract for the image-like feature-selection dataset."""

import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "feature_selection"))

from feature_selection_datasets import build_cnn_dataset  # noqa: E402


def test_shapes_and_loader():
    """Twenty 8x8 samples arrive in two batches of ten."""
    data = build_cnn_dataset()
    assert data.inputs.shape == (20, 64)
    assert data.targets.shape == (20,)
    assert data.targets.dtype == torch.long
    assert data.feature_dim == 64
    assert data.signal_features == [10, 45]
    batches = list(data.loader)
    assert [tuple(inputs.shape) for inputs, _ in batches] == [(10, 64), (10, 64)]


def test_labels_compare_the_two_signal_pixels():
    """The class is whether the first signal feature exceeds the second."""
    data = build_cnn_dataset()
    expected = (data.inputs[:, 10] > data.inputs[:, 45]).long()
    assert torch.equal(data.targets, expected)
