"""Pin the loss returned by FeatureSelectionWrapper.fit_weights."""

import torch
from torch import nn

from kaiwu.torch_plugin.maifs import FeatureSelectionWrapper


def _setup():
    torch.manual_seed(0)
    selector = FeatureSelectionWrapper(nn.Linear(4, 1), feature_dim=4)
    batches = [
        (torch.randn(4, 4), torch.randn(4, 1)),
        (torch.randn(2, 4), torch.randn(2, 1)),
    ]
    return selector, batches


def test_fit_weights_returns_the_batch_average_without_updates() -> None:
    selector, batches = _setup()
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        expected = (
            sum(float(loss_fn(selector(x), y)) for x, y in batches) / len(batches)
        )

    optimizer = torch.optim.SGD(selector.model.parameters(), lr=0.0)
    returned = selector.fit_weights(
        batches, loss_fn, optimizer, train_epochs=1
    )

    assert abs(returned - expected) < 1e-6


def test_fit_weights_averages_over_epochs() -> None:
    selector, batches = _setup()
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        per_epoch = sum(float(loss_fn(selector(x), y)) for x, y in batches) / len(batches)

    optimizer = torch.optim.SGD(selector.model.parameters(), lr=0.0)
    returned = selector.fit_weights(
        batches, loss_fn, optimizer, train_epochs=3
    )

    assert abs(returned - per_epoch) < 1e-6
