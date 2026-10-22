"""Pin the local-search iteration-budget validation."""

import numpy as np
import pytest

from kaiwu.torch_plugin.maifs.qubo import solve_qubo

QUBO = np.array([[1.0, 0.0], [0.0, 1.0]])
LINEAR = np.zeros(2)
INITIAL = np.zeros(2, dtype=int)


def test_zero_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        solve_qubo(QUBO, LINEAR, INITIAL, "local_search", max_iter=0)


def test_negative_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        solve_qubo(QUBO, LINEAR, INITIAL, "local_search", max_iter=-5)


def test_positive_budget_still_solves() -> None:
    solution = solve_qubo(QUBO, LINEAR, INITIAL, "local_search", max_iter=1)
    assert solution.shape == (2,)
    assert set(np.unique(solution)).issubset({0, 1})
