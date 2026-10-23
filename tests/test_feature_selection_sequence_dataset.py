"""Contract for the sequence feature-selection dataset."""

import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "feature_selection"))

from feature_selection_datasets import build_sequence_dataset  # noqa: E402


def test_shapes_and_loader():
    """Twenty sequences of length six with eight channels are batched by ten."""
    data = build_sequence_dataset()
    assert data.inputs.shape == (20, 6, 8)
    assert data.targets.shape == (20,)
    assert data.targets.dtype == torch.long
    assert data.feature_dim == 8
    assert data.signal_features == [1, 6]
    batches = list(data.loader)
    assert [tuple(inputs.shape) for inputs, _ in batches] == [(10, 6, 8), (10, 6, 8)]


def test_labels_compare_the_time_averaged_signal_channels():
    """The class is the sign of the mean difference over the sequence."""
    data = build_sequence_dataset()
    score = data.inputs[:, :, 1].mean(dim=1) - data.inputs[:, :, 6].mean(dim=1)
    assert torch.equal(data.targets, (score > 0).long())
