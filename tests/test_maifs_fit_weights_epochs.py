"""Regression test for ``FeatureSelectionWrapper.fit_weights`` argument validation.

``fit_weights(..., train_epochs=0)`` did not run the training loop, left
``batch_count`` at zero and then raised the *empty-loader* error on a perfectly
valid loader:

    ValueError: data_loader produced no batches

The message is factually wrong: the loader had batches, no epochs were
requested.  A zero-epoch call must be rejected with a message that names the
argument, while a genuinely empty loader keeps its own error.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from kaiwu.torch_plugin import FeatureSelectionWrapper  # noqa: E402


def _wrapper(model=None):
    model = model if model is not None else nn.Linear(3, 1)
    torch.manual_seed(0)
    return FeatureSelectionWrapper(model, feature_dim=3)


def _loader(n=16):
    generator = torch.Generator().manual_seed(0)
    return DataLoader(
        TensorDataset(torch.randn(n, 3, generator=generator), torch.randn(n, 1, generator=generator)),
        batch_size=8,
    )


def test_zero_epochs_are_rejected_with_a_clear_message():
    wrapper = _wrapper()
    optimizer = torch.optim.SGD(wrapper.model.parameters(), lr=0.01)

    with pytest.raises(ValueError) as excinfo:
        wrapper.fit_weights(_loader(), nn.MSELoss(), optimizer, train_epochs=0)

    message = str(excinfo.value)
    assert "train_epochs" in message
    assert "no batches" not in message


def test_empty_loader_keeps_its_own_error():
    wrapper = _wrapper()
    optimizer = torch.optim.SGD(wrapper.model.parameters(), lr=0.01)
    empty = DataLoader(TensorDataset(torch.empty(0, 3), torch.empty(0, 1)), batch_size=8)

    with pytest.raises(ValueError, match="no batches"):
        wrapper.fit_weights(empty, nn.MSELoss(), optimizer, train_epochs=1)


def test_one_epoch_still_trains_and_returns_a_loss():
    wrapper = _wrapper()
    optimizer = torch.optim.SGD(wrapper.model.parameters(), lr=0.01)

    loss = wrapper.fit_weights(_loader(), nn.MSELoss(), optimizer, train_epochs=1)

    assert isinstance(loss, float)
    assert loss > 0.0
