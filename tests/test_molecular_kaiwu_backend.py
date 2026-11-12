"""Regression tests for the molecular-design Kaiwu SA backend."""

import os
import sys
import unittest
from unittest import mock

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
sys.path.insert(
    0,
    os.path.join(REPO_ROOT, 'example', 'molecular_design', 'cim_sa'),
)

import kaiwu as kw  # noqa: E402  pylint: disable=wrong-import-position
from qc_mol_design.optimization import qubo as molecular_qubo  # noqa: E402


class _FakeSimulatedAnnealingOptimizer:
    """Record the solve call and return a fixed spin pool."""

    last_call = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def solve(self, ising_matrix=None, negtail_flip=True, sort_solutions=False):
        type(self).last_call = {
            'num_columns': len(ising_matrix),
            'nettail_flip': negtail_flip,
            'sort_solutions': sort_solutions,
        }
        return np.ones((1, len(ising_matrix)))


class TestMolecularKaiwuBackend(unittest.TestCase):
    """The documented ``QC_QUBO_SOLVER=kaiwu`` arm must run on Kaiwu 1.4.x."""

    def test_solve_qubo_uses_supported_converter_and_solve_call(self):
        matrix = np.array([[1.0, 0.5], [0.0, -2.0]])
        _FakeSimulatedAnnealingOptimizer.last_call = None
        with mock.patch.dict(os.environ, {'QC_QUBO_SOLVER': 'kaiwu'}):
            with mock.patch.object(
                molecular_qubo,
                '_ensure_kaiwu_license',
                lambda: None,
            ):
                with mock.patch.object(
                    kw.classical,
                    'SimulatedAnnealingOptimizer',
                    _FakeSimulatedAnnealingOptimizer,
                ):
                    solutions = molecular_qubo.solve_qubo(
                        matrix,
                        num_reads=1,
                        num_sweeps=10,
                    )

        self.assertIsInstance(solutions, list)
        self.assertEqual(len(solutions), 1)
        self.assertEqual(set(solutions[0]), {0, 1})
        self.assertEqual(
            _FakeSimulatedAnnealingOptimizer.last_call['num_columns'],
            matrix.shape[0] + 1,
        )

    def test_warm_start_fails_with_an_actionable_message(self):
        matrix = np.array([[1.0, 0.5], [0.0, -2.0]])
        with mock.patch.dict(os.environ, {'QC_QUBO_SOLVER': 'kaiwu'}):
            with mock.patch.object(
                molecular_qubo,
                '_ensure_kaiwu_license',
                lambda: None,
            ):
                with mock.patch.object(
                    kw.classical,
                    'SimulatedAnnealingOptimizer',
                    _FakeSimulatedAnnealingOptimizer,
                ):
                    with self.assertRaisesRegex(RuntimeError, 'warm starts'):
                        molecular_qubo.solve_qubo(
                            matrix,
                            num_reads=1,
                            num_sweeps=10,
                            initial_states=[{0: 1, 1: 0}],
                        )


if __name__ == '__main__':
    unittest.main()
