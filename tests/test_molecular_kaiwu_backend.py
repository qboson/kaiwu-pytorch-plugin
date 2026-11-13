"""Regression tests for the molecular-design Kaiwu backend on SDK 1.4.x."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
sys.path.insert(
    0,
    os.path.join(REPO_ROOT, 'example', 'molecular_design', 'cim_sa'),
)

import kaiwu as kw  # noqa: E402  pylint: disable=wrong-import-position
from qc_mol_design.optimization import cim_hardware  # noqa: E402
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


class _FakeCIMOptimizer:
    """Kaiwu 1.4.x-shaped CIM optimizer that returns a valid spin pool."""

    def __init__(
        self,
        task_name,
        wait=False,
        interval=1,
        project_no=None,
        task_mode='optimization',
        sample_number=10,
    ):
        self.task_name = task_name

    def solve(self, ising_matrix=None, negtail_flip=True, sort_solutions=False):
        del negtail_flip, sort_solutions
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

    def test_cim_license_initialization_survives_missing_private_adapter(self):
        """Kaiwu 1.4.x has no ``kaiwu.cim._optimizer_adapter`` module."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            credentials = root / 'credentials.json'
            credentials.write_text(
                json.dumps({'user_id': 'user', 'sdk_code': 'code'}),
                encoding='utf-8',
            )
            license_path = root / 'cim_license.lic'
            license_path.write_text('placeholder', encoding='utf-8')
            with mock.patch.dict(
                os.environ,
                {'KAIWU_CREDENTIALS_FILE': str(credentials)},
            ):
                with mock.patch.object(cim_hardware, '_LICENSE_READY', False):
                    with mock.patch(
                        'kaiwu.license._license_utils._read_license',
                        side_effect=OSError('missing'),
                    ):
                        with mock.patch('kaiwu.license.init') as init:
                            cim_hardware.initialize_license()
        init.assert_called_once_with('user', 'code')

    def test_cim_solver_construction_omits_removed_option(self):
        """Kaiwu 1.4.x rejects ``sample_sort_mode`` in the constructor."""
        matrix = np.array([[1.0, 0.5], [0.0, -2.0]])
        with tempfile.TemporaryDirectory() as directory:
            environment = {
                'KAIWU_CIM_PROJECT_NO': 'project',
                'KAIWU_CIM_RUN_ID': 'run',
                'KAIWU_CIM_TASK_CONTEXT': 'context',
                'KAIWU_CIM_CACHE_DIR': directory,
            }
            with mock.patch.dict(os.environ, environment):
                with mock.patch.object(
                    cim_hardware,
                    'initialize_license',
                    lambda: kw,
                ):
                    with mock.patch.object(
                        kw.cim,
                        'CIMOptimizer',
                        _FakeCIMOptimizer,
                    ):
                        solutions = cim_hardware.solve_hardware(
                            matrix,
                            num_reads=10,
                        )

        self.assertIsInstance(solutions, list)
        self.assertEqual(len(solutions), 1)
        self.assertEqual(set(solutions[0]), {0, 1})


if __name__ == '__main__':
    unittest.main()
