"""SQA budgets must not collide across experiment working directories."""
import importlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest


@pytest.fixture
def solver(monkeypatch):
    prefix = "qc_mol_design"
    saved = {k: v for k, v in sys.modules.items()
             if k == prefix or k.startswith(prefix + ".")}
    for name in saved:
        sys.modules.pop(name)
    root = Path(__file__).resolve().parents[1] / "example/molecular_design/sqa"
    monkeypatch.syspath_prepend(str(root))
    try:
        with warnings.catch_warnings():
            yield importlib.import_module("qc_mol_design.optimization.sqa_solver")
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                sys.modules.pop(name)
        sys.modules.update(saved)


def budget(seed, repeats):
    return {"steps": [{"global_step": 0, "solver_seed": seed,
                       "solver_repeats": repeats, "feasibility_batches": 1,
                       "candidate_reads": 10 * repeats,
                       "sampling_regime": "test"}]}


@pytest.mark.parametrize("first,second", [(1, 5), (5, 1)])
def test_relative_budgets_follow_working_directory(solver, monkeypatch, tmp_path, first, second):
    monkeypatch.setenv("QC_QUBO_SOLVER", "sqa")
    monkeypatch.setenv("QC_SPIN_ENCODING", "gauge")
    monkeypatch.setenv("QC_SQA_STEP", "0")
    monkeypatch.setenv("QC_SQA_BUDGET_FILE", "budget.json")
    # Sampling itself is outside this cache-routing test; all encoding, decoding,
    # file caching and candidate-count checks execute the production code.
    calls = []

    def sample(matrix, reads, sweeps, seed):
        calls.append(seed)
        return np.ones((reads, len(matrix)), dtype=np.int8), np.ones(sweeps), np.zeros(sweeps)

    monkeypatch.setattr(solver, "sample_physical", sample)
    prepared = {"ising": np.zeros((2, 2), dtype=np.int64), "metadata": {
        "precision_policy": "test", "structural_integer_exact": True, "penalty_ratio": 4.}}
    for name, repeats in (("first", first), ("second", second)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "budget.json").write_text(json.dumps(budget(solver.CFG.seed, repeats)))
        monkeypatch.chdir(directory)
        monkeypatch.setenv("QC_SQA_CACHE_DIR", str(directory / "cache"))
        result = solver.solve_budgeted(np.zeros((1, 1)), 10, 1000, prepared)
        assert len(result) == 10 * repeats
        assert solver.LAST_CALL["solver_repeats"] == repeats
        assert solver.LAST_CALL["sqa_calls"] == repeats
    assert len(calls) == first + second
    # Resolving to an absolute alias should reuse the same parsed budget and
    # the disk sample cache rather than creating a third cache identity.
    monkeypatch.setenv("QC_SQA_BUDGET_FILE", str(directory / "budget.json"))
    result = solver.solve_budgeted(np.zeros((1, 1)), 10, 1000, prepared)
    assert len(result) == 10 * second
    assert len(solver._BUDGETS) == 2
    assert len(calls) == first + second
    assert solver.LAST_CALL["sqa_cache_reused"]
