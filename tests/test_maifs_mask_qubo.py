"""Inspect the same objective that MAIFS sends to its configured solver."""

import numpy as np
import torch
from torch import nn

from kaiwu.torch_plugin import FeatureSelectionWrapper
from kaiwu.torch_plugin.maifs import plugin


def test_build_mask_qubo_returns_the_actual_objective_without_solving(monkeypatch):
    layer = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0]]))
    wrapper = FeatureSelectionWrapper(layer, 2, lambda_reg=0.2)
    samples = [(torch.eye(2), torch.tensor([[1.0], [0.0]]))]
    calls = []

    def solve(quadratic, linear, initial, solver, **kwargs):
        calls.append((quadratic.copy(), linear.copy()))
        return np.array([1, 0])

    monkeypatch.setattr(plugin, "solve_qubo", solve)
    quadratic, linear = wrapper.build_mask_qubo(samples, nn.MSELoss())

    assert not calls
    np.testing.assert_array_equal(wrapper.get_support(), [True, True])
    np.testing.assert_allclose(quadratic, np.diag([1.0, 4.0]))
    np.testing.assert_allclose(linear, [-0.8, 0.2])
    np.testing.assert_array_equal(wrapper.update_mask(samples, nn.MSELoss()), [1, 0])
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], quadratic)
    np.testing.assert_array_equal(calls[0][1], linear)
