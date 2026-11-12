"""Regression test for the ``sqa`` copy of the molecular Kaiwu backend."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SQA_SCRIPT = r"""
import os
import numpy as np
import kaiwu as kw
from qc_mol_design.optimization import qubo as molecular_qubo


class FakeSA:
    def __init__(self, *args, **kwargs):
        pass

    def solve(self, matrix=None, **kwargs):
        return np.ones((1, len(matrix)))


kw.classical.SimulatedAnnealingOptimizer = FakeSA
molecular_qubo._ensure_kaiwu_license = lambda: None
os.environ["QC_QUBO_SOLVER"] = "kaiwu"
matrix = np.array([[1.0, 0.5], [0.0, -2.0]])
solutions = molecular_qubo.solve_qubo(matrix, num_reads=1, num_sweeps=10)
assert isinstance(solutions, list) and len(solutions) == 1
assert set(solutions[0]) == {0, 1}
try:
    molecular_qubo.solve_qubo(
        matrix, num_reads=1, num_sweeps=10, initial_states=[{0: 1, 1: 0}]
    )
except RuntimeError as error:
    assert "warm starts" in str(error)
else:
    raise AssertionError("warm start unexpectedly succeeded")
"""


class TestSqaVariantBackend(unittest.TestCase):
    def test_sqa_variant_uses_the_fixed_backend_contract(self):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [
                str(ROOT / "example" / "molecular_design" / "sqa"),
                str(ROOT / "src"),
                environment.get("PYTHONPATH", ""),
            ]
        )
        result = subprocess.run(
            [sys.executable, "-c", SQA_SCRIPT],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
