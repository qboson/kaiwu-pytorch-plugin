"""Solver-solution decoding contract for the MAIFS QUBO helpers."""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from kaiwu.torch_plugin.maifs.qubo import _binary_from_solver_solution  # noqa: E402


def test_decoding_matches_the_qubo_bit_order_with_the_auxiliary_spin():
    """The aux spin folds the last variable back onto the QUBO bits."""
    binary = _binary_from_solver_solution(np.array([1.0, -1.0, -1.0]), 2)
    assert binary.tolist() == [0, 1]


def test_first_row_is_used_when_several_solutions_are_returned():
    """A solver returning multiple rows is decoded from the first row only."""
    solutions = np.array([[1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]])
    binary = _binary_from_solver_solution(solutions, 2)
    assert binary.tolist() == [1, 0]


def test_malformed_solutions_are_rejected():
    """Wrong length, too many dimensions and non-spin values all fail."""
    with pytest.raises(RuntimeError, match="auxiliary|dimension|\\-1/1|at least one"):
        _binary_from_solver_solution(np.array([1.0, 1.0]), 2)
    with pytest.raises(RuntimeError, match="auxiliary|dimension|\\-1/1|at least one"):
        _binary_from_solver_solution(np.zeros((2, 2, 2)), 2)
    with pytest.raises(RuntimeError, match="auxiliary|dimension|\\-1/1|at least one"):
        _binary_from_solver_solution(np.array([1.0, 0.0, 1.0]), 2)
