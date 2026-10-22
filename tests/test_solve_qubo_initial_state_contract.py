"""solve_qubo must honor the caller's initial state when no move improves."""

import numpy as np

from kaiwu.torch_plugin.maifs.qubo import solve_qubo


def test_degenerate_qubo_returns_the_initial_state() -> None:
    qubo = np.zeros((3, 3))
    linear = np.zeros(3)
    initial = np.array([1, 0, 1])

    solution = solve_qubo(qubo, linear, initial, "local_search")

    assert np.array_equal(solution, initial)
