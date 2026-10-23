"""Local-search Ising contract for the MAIFS QUBO helpers."""

import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from kaiwu.torch_plugin.maifs.qubo import _solve_ising_local_search  # noqa: E402


def ferromagnetic_matrix():
    """Three spins with equal negative couplings; the minimum is all-equal spins."""
    matrix = -np.ones((3, 3))
    np.fill_diagonal(matrix, 0.0)
    return matrix


def objective(matrix, spins):
    """Upper-triangle Ising objective used by the local search."""
    return float(np.sum(np.triu(matrix) * np.outer(spins, spins)))


def test_local_search_reaches_the_global_minimum():
    """From the all-ones start the greedy flip keeps the aligned state."""
    matrix = ferromagnetic_matrix()
    solution = _solve_ising_local_search(matrix)
    assert solution.shape == (1, 3)
    spins = solution.flatten()
    assert set(spins.tolist()) <= {-1, 1}
    assert objective(matrix, spins) == -3.0


def test_initial_binary_state_is_mapped_onto_the_spins():
    """A zero matrix leaves the initial binary state untouched."""
    solution = _solve_ising_local_search(
        np.zeros((3, 3)), initial_binary=np.array([1, 0])
    )
    assert solution.tolist() == [[1, -1, 1]]
