"""fit_weights must call update_mask at the configured interval."""

import numpy as np
import torch
from torch import nn

from kaiwu.torch_plugin.maifs import FeatureSelectionWrapper


def _run(interval):
    torch.manual_seed(0)
    selector = FeatureSelectionWrapper(
        nn.Linear(4, 1), feature_dim=4, mask_update_epochs=interval
    )
    calls = []

    def fake_update_mask(data_loader, loss_fn):
        del data_loader, loss_fn
        calls.append(1)
        return np.ones(4, dtype=int)

    selector.update_mask = fake_update_mask
    inputs = torch.randn(4, 4)
    targets = torch.randn(4, 1)
    optimizer = torch.optim.SGD(selector.model.parameters(), lr=0.0)
    selector.fit_weights([(inputs, targets), (inputs, targets)], nn.MSELoss(), optimizer, 2)
    return len(calls)


def test_interval_of_one_updates_every_epoch() -> None:
    assert _run(1) == 2


def test_no_interval_skips_updates() -> None:
    assert _run(None) == 0
