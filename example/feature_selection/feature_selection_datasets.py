"""Synthetic fixtures for solver comparisons.

Pass ``seed=42`` to a builder to reproduce inputs, labels and planted signals
independently of model initialization. Omitting it keeps random datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class FeatureSelectionDataset:
    """Container for one feature-selection example dataset."""

    loader: DataLoader
    inputs: torch.Tensor
    targets: torch.Tensor
    signal_features: list[int]
    feature_dim: int


def _torch_generator(seed: int | None) -> torch.Generator:
    """Create a local RNG; None keeps nondeterministic dataset generation."""
    generator = torch.Generator()
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    return generator


def build_linear_regression_dataset(*, seed: int | None = None) -> FeatureSelectionDataset:
    """Build regression data; seed controls planted signals, inputs and noise locally."""
    sample_count = 1024
    feature_dim = 100
    signal_count = 5

    rng = np.random.default_rng(seed)
    generator = _torch_generator(seed)
    signal_indices = rng.choice(feature_dim, size=signal_count, replace=False)
    signal_weights = rng.uniform(5.0, 10.0, size=signal_count)
    signal_weights *= rng.choice([-1.0, 1.0], size=signal_count)

    inputs = torch.randn(sample_count, feature_dim, generator=generator)
    true_weight = torch.zeros(feature_dim, 1)
    true_weight[torch.tensor(signal_indices, dtype=torch.long)] = torch.tensor(
        signal_weights,
        dtype=true_weight.dtype,
    ).view(-1, 1)
    targets = inputs @ true_weight + 0.001 * torch.randn(sample_count, 1, generator=generator)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=128, shuffle=False)
    return FeatureSelectionDataset(
        loader=loader,
        inputs=inputs,
        targets=targets,
        signal_features=sorted(signal_indices.tolist()),
        feature_dim=feature_dim,
    )


def build_cnn_dataset(*, seed: int | None = None) -> FeatureSelectionDataset:
    """Build image-like data with an optional local seed, leaving global RNGs intact."""
    sample_count = 20
    image_size = 8
    feature_dim = image_size * image_size
    signal_features = [10, 45]

    inputs = torch.randn(sample_count, feature_dim, generator=_torch_generator(seed))
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


def build_sequence_dataset(*, seed: int | None = None) -> FeatureSelectionDataset:
    """Build sequence data with an optional local seed, leaving global RNGs intact."""
    sample_count = 20
    sequence_length = 6
    feature_dim = 8
    signal_features = [1, 6]

    inputs = torch.randn(sample_count, sequence_length, feature_dim,
                         generator=_torch_generator(seed))
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
