"""fit_weights must reject a non-positive training-epoch count."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from kaiwu.torch_plugin import FeatureSelectionWrapper


def _setup():
    selector = FeatureSelectionWrapper(
        torch.nn.Linear(4, 1),
        feature_dim=4,
        min_selected_features=1,
        max_selected_features=3,
    )
    loader = DataLoader(
        TensorDataset(torch.randn(8, 4), torch.randn(8, 1)), batch_size=4
    )
    optimizer = torch.optim.SGD(selector.parameters(), lr=0.1)
    return selector, loader, optimizer


@pytest.mark.parametrize("epochs", [0, -1])
def test_non_positive_train_epochs_is_rejected(epochs: int) -> None:
    selector, loader, optimizer = _setup()
    with pytest.raises(ValueError, match="train_epochs must be a positive integer"):
        selector.fit_weights(loader, torch.nn.MSELoss(), optimizer, train_epochs=epochs)


@pytest.mark.parametrize("epochs", [1.5, True])
def test_non_integer_train_epochs_is_rejected(epochs) -> None:
    selector, loader, optimizer = _setup()
    with pytest.raises(TypeError, match="train_epochs must be a positive integer"):
        selector.fit_weights(loader, torch.nn.MSELoss(), optimizer, train_epochs=epochs)


def test_one_epoch_still_trains() -> None:
    selector, loader, optimizer = _setup()
    loss = selector.fit_weights(loader, torch.nn.MSELoss(), optimizer, train_epochs=1)
    assert loss > 0.0
