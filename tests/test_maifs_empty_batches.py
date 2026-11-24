"""Empty MAIFS batches must never act as training or derivative samples."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn

from kaiwu.torch_plugin import FeatureSelectionWrapper
from kaiwu.torch_plugin.maifs import plugin


def make_selector():
    """Use known initial weights and nonzero decay to reveal phantom updates."""
    layer = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.fill_(0.5)
    selector = FeatureSelectionWrapper(layer, 2)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.2, weight_decay=0.5)
    return selector, optimizer


def test_fit_with_only_empty_batches_does_not_touch_weights():
    """An all-empty loader has no optimization data."""
    selector, optimizer = make_selector()
    before = selector.model.weight.detach().clone()
    with pytest.raises(ValueError, match="no batches with samples"):
        selector.fit_weights([(torch.empty(0, 2), torch.empty(0, 1))], nn.MSELoss(), optimizer)
    torch.testing.assert_close(selector.model.weight, before, atol=0, rtol=0)


def test_empty_batches_do_not_change_training_loss_or_update():
    """A filtering loader may leave empty batches between ordinary rows."""
    batch = (torch.tensor([[1.0, 2.0], [2.0, 1.0]]), torch.tensor([[0.0], [1.0]]))
    empty = (torch.empty(0, 2), torch.empty(0, 1))
    expected, reference_optimizer = make_selector()
    actual, optimizer = make_selector()
    loss_fn = nn.MSELoss()

    expected_loss = expected.fit_weights([batch], loss_fn, reference_optimizer)
    actual_loss = actual.fit_weights([empty, batch, empty], loss_fn, optimizer)

    assert actual_loss == expected_loss
    torch.testing.assert_close(actual.model.weight, expected.model.weight, atol=0, rtol=0)


def test_mask_update_with_no_samples_does_not_call_the_solver(monkeypatch):
    """An all-empty loader cannot supply derivatives to a mask update."""
    selector, _ = make_selector()
    solver = Mock(return_value=[1, 0])
    monkeypatch.setattr(plugin, "solve_qubo", solver)
    with pytest.raises(ValueError, match="no batches with samples"):
        selector.update_mask([(torch.empty(0, 2), torch.empty(0, 1))], nn.MSELoss())
    solver.assert_not_called()
    assert selector.selected_indices().tolist() == [0, 1]


def test_derivatives_use_nonempty_batches_only():
    """The same observed rows yield the same loss gradient and Hessian."""
    selector, _ = make_selector()
    batch = (torch.tensor([[1.0, 2.0], [2.0, 1.0]]), torch.tensor([[0.0], [1.0]]))
    empty = (torch.empty(0, 2), torch.empty(0, 1))
    expected = selector.compute_mask_derivatives([batch], nn.MSELoss())
    actual = selector.compute_mask_derivatives([empty, batch, empty], nn.MSELoss())
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result, reference)
