"""MAIFS must keep the best row when a solver returns several solutions."""

import numpy as np
import pytest

import kaiwu as kw
from kaiwu.torch_plugin.maifs import qubo


def test_cim_keeps_best_of_several_solutions(monkeypatch, tmp_path):
    """CIM multi-row output must not be reduced to the first row."""

    class MultiRowCIM:
        """Return a bad row first and the optimum second."""

        def __init__(self, **kwargs):
            pass

        def solve(self, matrix):
            return np.array([[1.0, 1.0], [1.0, -1.0]])

    monkeypatch.setattr(kw.cim, "CIMOptimizer", MultiRowCIM)
    monkeypatch.setattr(kw.common.CheckpointManager, "save_dir", None)

    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = qubo._solve_ising_kaiwu_cim(
        matrix,
        target_precision=8,
        max_precision=8,
        max_bits=20,
        sample_number=2,
        save_dir=tmp_path,
        cleanup_records=True,
    )

    spins = np.asarray(result)[0]
    assert spins @ matrix @ spins == pytest.approx(-2.0)


def test_sa_keeps_best_of_several_solutions(monkeypatch):
    """SA multi-row output must be decoded from the best row."""

    class MultiRowSA:
        """Return a bad row first and the optimum second."""

        def __init__(self, **kwargs):
            pass

        def solve(self, matrix):
            return np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 1.0]])

    monkeypatch.setattr(kw.classical, "SimulatedAnnealingOptimizer", MultiRowSA)

    quadratic = np.array([[0.0, 1.0], [1.0, 0.0]])
    selected = qubo.solve_qubo(
        quadratic_matrix=quadratic,
        linear_vector=np.zeros(2),
        initial_state=np.zeros(2, dtype=int),
        solver="sa",
        size_limit=2,
    )

    assert 0.5 * selected @ quadratic @ selected == pytest.approx(0.0)