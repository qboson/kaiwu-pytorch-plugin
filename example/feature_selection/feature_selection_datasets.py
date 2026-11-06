from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# Fixed seed so every synthetic dataset is identical on each run.
DATASET_SEED = 2026


def _seeded_generator() -> torch.Generator:
    return torch.Generator().manual_seed(DATASET_SEED)


@dataclass(frozen=True)
class FeatureSelectionDataset:
    """Container for one feature-selection example dataset."""

    loader: DataLoader
    inputs: torch.Tensor
    targets: torch.Tensor
    signal_features: list[int]
    feature_dim: int


def build_linear_regression_dataset() -> FeatureSelectionDataset:
    """Build a synthetic linear-regression feature-selection dataset."""
    sample_count = 1024
    feature_dim = 100
    signal_count = 5

    rng = np.random.default_rng(DATASET_SEED)
    signal_indices = rng.choice(feature_dim, size=signal_count, replace=False)
    signal_weights = rng.uniform(5.0, 10.0, size=signal_count)
    signal_weights *= rng.choice([-1.0, 1.0], size=signal_count)

    inputs = torch.randn(sample_count, feature_dim, generator=_seeded_generator())
    true_weight = torch.zeros(feature_dim, 1)
    true_weight[torch.tensor(signal_indices, dtype=torch.long)] = torch.tensor(
        signal_weights,
        dtype=true_weight.dtype,
    ).view(-1, 1)
    targets = inputs @ true_weight + 0.001 * torch.randn(
        sample_count, 1, generator=_seeded_generator()
    )
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=128, shuffle=False)
    return FeatureSelectionDataset(
        loader=loader,
        inputs=inputs,
        targets=targets,
        signal_features=sorted(signal_indices.tolist()),
        feature_dim=feature_dim,
    )


def build_cnn_dataset() -> FeatureSelectionDataset:
    """Build a synthetic image-like classification dataset."""
    sample_count = 20
    image_size = 8
    feature_dim = image_size * image_size
    signal_features = [10, 45]

    inputs = torch.randn(sample_count, feature_dim, generator=_seeded_generator())
    score = inputs[:, signal_features[0]] - inputs[:, signal_features[1]]
    targets = (score > 0).long()
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=10, shuffle=False)
    return FeatureSelectionDataset(
        loader=loader,
        inputs=inputs,
        targets=targets,
        signal_features=signal_features,
        feature_dim=feature_dim,
    )


def build_sequence_dataset() -> FeatureSelectionDataset:
    """Build a synthetic sequence classification dataset."""
    sample_count = 20
    sequence_length = 6
    feature_dim = 8
    signal_features = [1, 6]

    inputs = torch.randn(
        sample_count, sequence_length, feature_dim, generator=_seeded_generator()
    )
    score = (
        inputs[:, :, signal_features[0]].mean(dim=1)
        - inputs[:, :, signal_features[1]].mean(dim=1)
    )
    targets = (score > 0).long()
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=10, shuffle=False)
    return FeatureSelectionDataset(
        loader=loader,
        inputs=inputs,
        targets=targets,
        signal_features=signal_features,
        feature_dim=feature_dim,
    )
