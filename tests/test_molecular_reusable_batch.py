"""Check that a CIM tuning batch reuses all repeats from one matching run."""

import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("rdkit")


@pytest.fixture(params=["sqa", "cim_sa"])
def tuning_solver(request, monkeypatch):
    root = Path(__file__).resolve().parents[1] / "example" / "molecular_design" / request.param
    package = "qc_mol_design"
    previous = {
        key: sys.modules.pop(key)
        for key in list(sys.modules)
        if key == package or key.startswith(f"{package}.")
    }
    monkeypatch.syspath_prepend(str(root))
    try:
        yield importlib.import_module(f"{package}.optimization.tuning_solver")
    finally:
        for key in list(sys.modules):
            if key == package or key.startswith(f"{package}."):
                sys.modules.pop(key)
        sys.modules.update(previous)


@pytest.fixture
def batch(tuning_solver, monkeypatch, tmp_path):
    monkeypatch.setenv("QC_FIRST_BATCH_REUSE_DIR", str(tmp_path))
    qubo = np.array([[2, -3], [-3, 1]])
    prepared = {"ising": np.diag([1, 2, 3]), "metadata": {}}
    seed = 17

    def record(run_id, repeat, *, matrix_matches=True):
        call = tmp_path / "calls" / f"{run_id}-{repeat}"
        call.mkdir(parents=True)
        meta = {"context": f"trial_rep{repeat}", "run_id": run_id, "status": "prepared"}
        (call / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        encoded = tuning_solver.encode_prepared(prepared, "permute", seed + repeat * 100000)
        submitted = tuning_solver.physical_matrix(encoded["ising"], encoded)
        np.savez(
            call / "input.npz", qubo=qubo,
            ising=submitted if matrix_matches else submitted + 1,
        )

    def resume():
        return tuning_solver.previous_batch(qubo, prepared, "trial", 3, "permute", seed)

    return record, resume


def test_resumes_one_complete_matching_run_with_other_records_present(batch, tmp_path):
    record, resume = batch
    for repeat in range(3):
        record("a_complete", repeat)
    record("b_partial", 0)
    record("c_different_matrix", 1, matrix_matches=False)

    assert resume() == (str(tmp_path), "a_complete")


def test_rejects_two_complete_matching_runs(batch):
    record, resume = batch
    for repeat in range(3):
        record("first", repeat)
        record("second", repeat)

    with pytest.raises(RuntimeError, match="Multiple matching run IDs"):
        resume()


def test_rejects_mixing_matching_repeats_from_different_runs(batch):
    record, resume = batch
    record("first", 0)
    record("second", 1)
    record("second", 2)

    with pytest.raises(RuntimeError, match="Mixed run IDs"):
        resume()


def test_returns_no_cache_if_a_repeat_does_not_match(batch):
    record, resume = batch
    record("first", 0)
    record("first", 1)
    record("first", 2, matrix_matches=False)

    assert resume() is None
