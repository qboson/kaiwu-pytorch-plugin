"""Check the exact MAIFS reference against an independent enumeration."""

from itertools import product

import numpy as np
import pytest
import torch
from torch import nn

from kaiwu.torch_plugin import FeatureSelectionWrapper
from kaiwu.torch_plugin.maifs.qubo import solve_qubo


@pytest.mark.parametrize("size", [3, 6, 11])
def test_exact_solver_minimizes_original_binary_objective(size):
    rng = np.random.default_rng(size)
    quadratic = rng.normal(size=(size, size))
    linear = rng.normal(size=size)
    candidates = np.array(list(product((0, 1), repeat=size)))
    objectives = 0.5 * ((candidates @ quadratic) * candidates).sum(axis=1) + candidates @ linear
    expected = candidates[np.argmin(objectives)]

    for initial in (np.zeros(size), np.ones(size)):
        actual = solve_qubo(quadratic, linear, initial, solver="exact")
        np.testing.assert_array_equal(actual, expected)


def test_exact_solver_handles_ties_and_enforces_its_limit():
    result = solve_qubo(np.zeros((4, 4)), np.zeros(4), np.ones(4), solver="exact")
    np.testing.assert_array_equal(result, np.zeros(4))
    with pytest.raises(ValueError, match="at most 16"):
        solve_qubo(np.zeros((17, 17)), np.ones(17), np.ones(17), solver="exact")


def test_feature_selection_can_use_exact_solver():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.)
    inputs = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    wrapper = FeatureSelectionWrapper(model, 2, cardinality_k=1, solver="exact")
    mask = wrapper.update_mask([(inputs, inputs[:, :1])], nn.MSELoss())
    np.testing.assert_array_equal(mask, [1, 0])
