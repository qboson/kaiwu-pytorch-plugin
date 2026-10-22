"""PrecisionSplitExplorer must reject invalid budgets, matrices, and solutions."""

from __future__ import annotations

import numpy as np
import pytest

from kaiwu.torch_plugin.maifs.qubo import PrecisionSplitExplorer

MATRIX = np.array(
    [
        [0.0, -0.25, 0.125],
        [-0.25, 0.0, 0.375],
        [0.125, 0.375, 0.0],
    ]
)


def _explorer(**kwargs) -> PrecisionSplitExplorer:
    params = {"target_precision": 4, "max_bits": 12, "max_precision": 8}
    params.update(kwargs)
    return PrecisionSplitExplorer(**params)


@pytest.mark.parametrize("max_bits", [0, -1])
def test_nonpositive_max_bits_is_rejected(max_bits: int) -> None:
    with pytest.raises(ValueError, match="max_bits must be a positive integer"):
        _explorer(max_bits=max_bits)


@pytest.mark.parametrize("max_bits", [1.5, True])
def test_non_integer_max_bits_is_rejected(max_bits) -> None:
    with pytest.raises(TypeError, match="max_bits must be a positive integer"):
        _explorer(max_bits=max_bits)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_search_rejects_non_finite_matrices(value: float) -> None:
    matrix = MATRIX.copy()
    matrix[0, 1] = value
    matrix[1, 0] = value
    with pytest.raises(ValueError, match="ising_matrix must contain only finite"):
        _explorer().search(matrix)


@pytest.mark.parametrize("offset", [-1, 1])
def test_restore_rejects_wrong_length_solutions(offset: int) -> None:
    explorer = _explorer()
    plan = explorer.search(MATRIX)
    with pytest.raises(ValueError, match="solution must have"):
        explorer.restore_solution(np.ones(plan.split_size + offset, dtype=int))


def test_correct_solution_length_round_trips() -> None:
    explorer = _explorer()
    plan = explorer.search(MATRIX)
    restored = np.asarray(explorer.restore_solution(np.ones(plan.split_size, dtype=int)))
    assert restored.shape == (MATRIX.shape[0],)
    assert np.array_equal(restored, np.ones(MATRIX.shape[0], dtype=int))
