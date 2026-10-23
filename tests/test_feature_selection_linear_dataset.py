"""Contract for the linear-regression feature-selection dataset."""

import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "feature_selection"))

from feature_selection_datasets import build_linear_regression_dataset  # noqa: E402


def test_shapes_loader_and_signal_bookkeeping():
    """The dataset reports its documented shapes, batches and signal columns."""
    data = build_linear_regression_dataset()
    assert data.inputs.shape == (1024, 100)
    assert data.targets.shape == (1024, 1)
    assert data.feature_dim == 100
    assert len(data.signal_features) == 5
    assert data.signal_features == sorted(set(data.signal_features))
    assert all(0 <= index < 100 for index in data.signal_features)
    batch = next(iter(data.loader))
    assert batch[0].shape == (128, 100)
    assert batch[1].shape == (128, 1)


def test_targets_are_driven_by_the_reported_signal_features():
    """Only the reported columns carry weight in the generated targets."""
    data = build_linear_regression_dataset()
    solution = torch.linalg.lstsq(data.inputs, data.targets).solution
    recovered = solution.abs().flatten().topk(5).indices.tolist()
    assert sorted(recovered) == data.signal_features
    residual = (data.inputs @ solution - data.targets).abs().max().item()
    assert residual < 0.01
