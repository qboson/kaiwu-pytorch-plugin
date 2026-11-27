"""Exercise custom binary QUBO solvers through the feature-selection wrapper."""

from itertools import product

import numpy as np
import pytest
import torch
from torch import nn

from kaiwu.torch_plugin.maifs import FeatureSelectionWrapper
from kaiwu.torch_plugin.maifs.qubo import solve_qubo


def test_update_mask_accepts_a_binary_solver_with_options():
    model = nn.Linear(3, 1, bias=False)
    nn.init.ones_(model.weight)

    def exact_solver(quadratic, linear, initial, *, expected_size):
        np.testing.assert_array_equal(initial, [1, 1, 1])
        assert linear.shape == (expected_size,)
        states = np.array(list(product((0, 1), repeat=expected_size)))
        energies = 0.5 * ((states @ quadratic) * states).sum(axis=1) + states @ linear
        return states[np.argmin(energies)]

    selector = FeatureSelectionWrapper(
        model, 3, solver=exact_solver, solver_kwargs={"expected_size": 3}
    )
    selected = selector.update_mask(
        [(torch.eye(3), torch.tensor([[1.0], [0.0], [1.0]]))],
        nn.MSELoss(),
    )

    np.testing.assert_array_equal(selected, [1, 0, 1])
    np.testing.assert_array_equal(selector.get_support(), [True, False, True])


def test_custom_solver_can_modify_its_inputs_without_affecting_the_caller():
    quadratic = np.eye(2)
    linear = np.ones(2)
    initial = np.ones(2, dtype=int)

    def mutate(quadratic_copy, linear_copy, initial_copy):
        quadratic_copy.fill(0)
        linear_copy.fill(0)
        initial_copy.fill(0)
        return initial_copy

    np.testing.assert_array_equal(solve_qubo(quadratic, linear, initial, mutate), [0, 0])
    np.testing.assert_array_equal(quadratic, np.eye(2))
    np.testing.assert_array_equal(linear, [1, 1])
    np.testing.assert_array_equal(initial, [1, 1])


@pytest.mark.parametrize("result, message", [
    ([[0, 1]], "shape"),
    ([0, 0.5], "binary"),
    ([0, np.nan], "binary"),
])
def test_invalid_custom_solver_result_keeps_the_previous_mask(result, message):
    selector = FeatureSelectionWrapper(
        nn.Linear(2, 1), 2, solver=lambda *args: np.asarray(result)
    )

    with pytest.raises(RuntimeError, match=message):
        selector.update_mask([(torch.eye(2), torch.zeros(2, 1))], nn.MSELoss())

    np.testing.assert_array_equal(selector.get_support(), [True, True])
