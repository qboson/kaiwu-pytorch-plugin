"""Container-consistency contract for the feature-selection datasets."""

import os
import sys
from dataclasses import FrozenInstanceError

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "feature_selection"))

from feature_selection_datasets import (  # noqa: E402
    build_cnn_dataset,
    build_linear_regression_dataset,
    build_sequence_dataset,
)


@pytest.mark.parametrize(
    "builder", [build_linear_regression_dataset, build_cnn_dataset, build_sequence_dataset]
)
def test_loader_batches_match_the_exposed_tensors(builder):
    """Re-iterating the loader reproduces the stored inputs and targets."""
    data = builder()
    inputs = torch.cat([batch[0] for batch in data.loader], dim=0)
    targets = torch.cat([batch[1] for batch in data.loader], dim=0)
    assert torch.equal(inputs, data.inputs)
    assert torch.equal(targets, data.targets)
    assert len(data.loader.dataset) == data.inputs.shape[0]
    assert data.signal_features == sorted(data.signal_features)
    assert all(0 <= index < data.feature_dim for index in data.signal_features)


def test_container_is_immutable():
    """The dataclass is frozen so callers cannot rewrite the feature layout."""
    data = build_cnn_dataset()
    with pytest.raises(FrozenInstanceError):
        data.feature_dim = 1
