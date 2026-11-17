"""Periodic feature selection can use the same derivative settings as update_mask."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kaiwu.torch_plugin.maifs import FeatureSelectionWrapper, plugin


def test_periodic_update_uses_diagonal_curvature_and_the_requested_sample_cap(monkeypatch):
    model = nn.Linear(2, 1, bias=False)
    nn.init.ones_(model.weight)
    wrapper = FeatureSelectionWrapper(model, 2, mask_update_epochs=2)
    loader = DataLoader(
        TensorDataset(torch.ones(3, 2), torch.zeros(3, 1)), batch_size=2, shuffle=False
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    batch_sizes = []
    solver_matrices = []

    def record_loss(output, target):
        batch_sizes.append(len(output))
        return torch.nn.functional.mse_loss(output, target)

    def solve(quadratic, linear, initial, solver, **kwargs):
        solver_matrices.append(quadratic.copy())
        return np.array([1, 0])

    monkeypatch.setattr(plugin, "solve_qubo", solve)
    wrapper.fit_weights(
        loader, record_loss, optimizer, train_epochs=2, hessian_mode="diagonal", max_samples=1
    )

    assert batch_sizes == [2, 1, 2, 1, 1]
    assert len(solver_matrices) == 1
    np.testing.assert_allclose(solver_matrices[0], np.diag([2.0, 2.0]))
    np.testing.assert_array_equal(wrapper.get_support(), [True, False])


@pytest.mark.parametrize("options", [{"hessian_mode": "unknown"}, {"max_samples": 0}])
def test_invalid_periodic_settings_raise_before_updating_model_weights(options):
    model = nn.Linear(2, 1, bias=False)
    nn.init.ones_(model.weight)
    wrapper = FeatureSelectionWrapper(model, 2, mask_update_epochs=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = [(torch.ones(1, 2), torch.zeros(1, 1))]

    with pytest.raises(ValueError, match=next(iter(options))):
        wrapper.fit_weights(data, nn.MSELoss(), optimizer, **options)
    torch.testing.assert_close(model.weight, torch.ones_like(model.weight))
