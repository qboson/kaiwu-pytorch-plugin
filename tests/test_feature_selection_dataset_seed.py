"""Synthetic comparisons must be reproducible without changing global RNGs."""
import numpy as np
import pytest
import torch

from example.feature_selection import feature_selection_datasets as datasets


@pytest.mark.parametrize("builder", [datasets.build_linear_regression_dataset,
                                     datasets.build_cnn_dataset,
                                     datasets.build_sequence_dataset])
def test_dataset_seed_is_local_and_repeatable(builder):
    torch.manual_seed(124)
    np.random.seed(124)
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    first = builder(seed=17)
    torch.rand(5)
    np.random.random(5)
    second = builder(seed=17)
    torch.testing.assert_close(first.inputs, second.inputs, rtol=0, atol=0)
    torch.testing.assert_close(first.targets, second.targets, rtol=0, atol=0)
    assert first.signal_features == second.signal_features
    assert not torch.equal(first.inputs, builder(seed=18).inputs)
    torch.set_rng_state(torch_before)
    np.random.set_state(numpy_before)
    builder(seed=17)
    assert torch.equal(torch_before, torch.get_rng_state())
    assert np.array_equal(numpy_before[1], np.random.get_state()[1])


def test_linear_targets_and_signal_features_agree():
    data = datasets.build_linear_regression_dataset(seed=3)
    # Recovering the planted coefficients independently verifies the fixture.
    weights = torch.linalg.lstsq(data.inputs, data.targets).solution[:, 0]
    assert set(torch.topk(weights.abs(), 5).indices.tolist()) == set(data.signal_features)
    recovered = weights[data.signal_features].abs()
    assert torch.all((recovered > 4.99) & (recovered < 10.01))
