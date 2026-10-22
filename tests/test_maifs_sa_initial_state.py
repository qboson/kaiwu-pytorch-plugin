"""Regression test for the MAIFS simulated-annealing warm start."""

import numpy as np
import pytest

import kaiwu as kw
from kaiwu.torch_plugin.maifs import qubo as maifs_qubo


class _RecordingSA:
    """Stands in for the license-gated Kaiwu SA optimizer."""

    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def solve(self, ising_matrix=None, init_solution=None):
        _RecordingSA.calls.append(init_solution)
        return np.array([1, -1, 1, 1])


def test_sa_solver_forwards_the_initial_state(monkeypatch):
    monkeypatch.setattr(kw.classical, "SimulatedAnnealingOptimizer", _RecordingSA)
    _RecordingSA.calls = []
    ising_matrix = np.zeros((4, 4))

    solution = maifs_qubo._solve_ising_sa(  # pylint: disable=protected-access
        ising_matrix, initial_binary=np.array([1, 0, 1])
    )

    assert len(_RecordingSA.calls) == 1
    assert np.array_equal(_RecordingSA.calls[0], np.array([1, -1, 1, 1]))
    assert solution.shape == (4,)


def test_sa_solver_without_initial_state_passes_none(monkeypatch):
    monkeypatch.setattr(kw.classical, "SimulatedAnnealingOptimizer", _RecordingSA)
    _RecordingSA.calls = []

    maifs_qubo._solve_ising_sa(np.zeros((4, 4)))  # pylint: disable=protected-access

    assert _RecordingSA.calls == [None]


def test_sa_solver_rejects_invalid_initial_state(monkeypatch):
    monkeypatch.setattr(kw.classical, "SimulatedAnnealingOptimizer", _RecordingSA)

    with pytest.raises(ValueError):
        maifs_qubo._solve_ising_sa(  # pylint: disable=protected-access
            np.zeros((4, 4)), initial_binary=np.array([1, 2, 3])
        )
