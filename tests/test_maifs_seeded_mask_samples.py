"""Verify reproducible population sampling for the MAIFS mask objective."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kaiwu.torch_plugin import FeatureSelectionWrapper


def observation_data():
    ids = torch.arange(42, dtype=torch.float32).view(-1, 1)
    return torch.cat((ids / 10, torch.ones_like(ids)), dim=1), ids


def test_seeded_derivatives_match_a_full_stream_reference():
    features, targets = observation_data()
    expected = np.sort(np.argsort(np.random.default_rng(7).random(len(targets)))[:6])
    assert max(expected) >= 12
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.2, -0.4]]))
    wrapper = FeatureSelectionWrapper(model, feature_dim=2)
    seen = []

    def capture_loss(prediction, target):
        seen.append(target.detach().flatten().tolist())
        return nn.functional.mse_loss(prediction, target)

    full_reference = [(features[expected], targets[expected])]
    reference = wrapper.compute_mask_derivatives(full_reference, nn.MSELoss(), max_samples=None)
    for batch_size in (1, 8, 19):
        loader = DataLoader(TensorDataset(features, targets), batch_size=batch_size)
        actual = wrapper.compute_mask_derivatives(loader, capture_loss, max_samples=6, sample_seed=7)
        assert seen[-1] == expected.tolist()
        for observed, wanted in zip(actual, reference):
            np.testing.assert_allclose(observed, wanted, rtol=1e-6, atol=1e-6)


def test_mask_update_uses_the_seed_and_unseeded_mode_keeps_the_prefix():
    features, targets = observation_data()
    batches = list(DataLoader(TensorDataset(features, targets), batch_size=12))
    wrapper = FeatureSelectionWrapper(nn.Linear(2, 1), 2)
    seen = []

    def capture_loss(prediction, target):
        seen.append(target.flatten().tolist())
        return nn.functional.mse_loss(prediction, target)

    wrapper.update_mask(batches, capture_loss, max_samples=4)
    wrapper.update_mask(batches, capture_loss, max_samples=4, sample_seed=7)
    assert seen[0] == [0, 1, 2, 3]
    assert len(seen[1]) == 4 and max(seen[1]) > 11


def test_seeded_subset_rejects_zero_budget_or_no_rows():
    wrapper = FeatureSelectionWrapper(nn.Linear(2, 1), 2)
    features, targets = observation_data()
    with pytest.raises(ValueError, match="positive"):
        wrapper.compute_mask_derivatives([(features, targets)], nn.MSELoss(),
                                         max_samples=0, sample_seed=2)
    with pytest.raises(ValueError, match="no observations"):
        wrapper.compute_mask_derivatives([(features[:0], targets[:0])], nn.MSELoss(),
                                         sample_seed=2)
