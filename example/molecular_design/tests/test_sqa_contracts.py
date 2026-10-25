"""Offline contracts and a small real SQA call; never initializes a CIM license."""
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sqa"))
import numpy as np
from qc_mol_design.config import CFG
from qc_mol_design.optimization import sqa_solver
from qc_mol_design.optimization.cim_precision import prepare_balanced_qubo
from qc_mol_design.optimization.spin_encoding import encode_prepared, physical_matrix, logical_spins


class SQAContracts(unittest.TestCase):
    def test_ising_sign_and_pair_factor_exhaustively(self):
        rng = np.random.default_rng(8)
        raw = rng.integers(-40, 40, (5, 5))
        matrix = np.triu(raw, 1)
        matrix = matrix + matrix.T
        bqm = sqa_solver.energy_model(matrix)
        prepared = encode_prepared({"ising": matrix, "metadata": {}}, "gauge", 42)
        physical = physical_matrix(matrix, prepared)
        physical_bqm = sqa_solver.energy_model(physical)
        for state in itertools.product((-1, 1), repeat=5):
            spins = np.asarray(state)
            self.assertEqual(bqm.energy(dict(enumerate(state))), -spins @ matrix @ spins)
            decoded = logical_spins(spins, prepared)[0]
            self.assertEqual(physical_bqm.energy(dict(enumerate(state))), -decoded @ matrix @ decoded)

    def test_transverse_schedule_and_real_pool(self):
        matrix = np.array([[0, 2, -1], [2, 0, 3], [-1, 3, 0]])
        spins, hp, hd = sqa_solver.sample_physical(matrix, 10, 100, 42)
        self.assertEqual(spins.shape, (10, 3))
        self.assertTrue(np.isin(spins, [-1, 1]).all())
        self.assertTrue(np.all(hd[:-1] > 0))
        self.assertEqual(hd[0], 2.0)
        self.assertEqual(hd[-1], 0.0)
        self.assertGreater(hp[-1], hp[0])

    def test_budget_and_cache_preserve_duplicate_reads_without_hardware(self):
        structure = np.array([[-1., 1.], [0., -1.]])
        Q, prepared = prepare_balanced_qubo(structure)
        def fake(matrix, reads, sweeps, seed):
            return np.ones((reads, len(matrix)), dtype=np.int8), np.ones(sweeps), np.ones(sweeps)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            budget = {"steps": [
                {"global_step": 0, "solver_seed": 42, "solver_repeats": 5,
                 "feasibility_batches": 2, "candidate_reads": 100,
                 "sampling_regime": "historical_5_calls"},
                {"global_step": 1, "solver_seed": 43, "solver_repeats": 1,
                 "feasibility_batches": 1, "candidate_reads": 10,
                 "sampling_regime": "single_call"}]}
            path = root / "budget.json"
            path.write_text(json.dumps(budget))
            with patch.dict(os.environ, {"QC_QUBO_SOLVER": "sqa", "QC_SPIN_ENCODING": "gauge",
                    "QC_SQA_BUDGET_FILE": str(path), "QC_SQA_CACHE_DIR": str(root / "cache"),
                    "QC_SQA_STEP": "0"}), patch.object(sqa_solver, "sample_physical", side_effect=fake) as sampler:
                CFG.seed = 42
                first = sqa_solver.solve_budgeted(Q, 10, 1000, prepared)
                self.assertEqual(len(first), 100)
                self.assertEqual(sampler.call_count, 10)
                again = sqa_solver.solve_budgeted(Q, 10, 1000, prepared)
                self.assertEqual(first, again)
                self.assertEqual(sampler.call_count, 10)
                self.assertFalse(sqa_solver.LAST_CALL["used_cim"])
                with self.assertRaises(AssertionError):
                    sqa_solver.solve_budgeted(Q + np.eye(2), 10, 1000, prepared)
                os.environ["QC_SQA_STEP"] = "1"
                CFG.seed = 43
                self.assertEqual(len(sqa_solver.solve_budgeted(Q, 10, 1000, prepared)), 10)
                self.assertEqual(sampler.call_count, 11)
                self.assertEqual(sqa_solver.LAST_CALL["sampling_regime"], "single_call")


if __name__ == "__main__":
    unittest.main()
