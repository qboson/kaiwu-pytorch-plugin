"""Regression tests for molecular_design example hardening.

Covers two behaviours:
  * QC_SQA_BUDGET_FILE is canonicalised and existence-checked before it is
    hashed into run metadata;
  * the ring warm-start initial states stay reproducible under the run's
    per-condition seeding protocol while drawing from a dedicated NumPy
    generator instead of the global random stream.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
SQA_ROOT = ROOT / "example/molecular_design/sqa"
SCRIPT = SQA_ROOT / "run_generation_strict.py"


def _load_generation_module():
    pytest.importorskip("pandas")
    if str(SQA_ROOT) not in sys.path:
        sys.path.insert(0, str(SQA_ROOT))
    spec = importlib.util.spec_from_file_location("run_generation_strict", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_budget_sha256_hashes_the_requested_file(tmp_path, monkeypatch):
    module = _load_generation_module()
    budget = tmp_path / "budget.json"
    budget.write_text('{"steps": []}', encoding="utf-8")
    monkeypatch.setenv("QC_SQA_BUDGET_FILE", str(budget))
    expected = hashlib.sha256(budget.read_bytes()).hexdigest()
    assert module.validated_budget_sha256() == expected


def test_budget_sha256_resolves_symlink_to_real_file(tmp_path, monkeypatch):
    module = _load_generation_module()
    budget = tmp_path / "budget.json"
    budget.write_text("{}", encoding="utf-8")
    link = tmp_path / "budget_link.json"
    link.symlink_to(budget)
    expected = hashlib.sha256(budget.read_bytes()).hexdigest()
    monkeypatch.setenv("QC_SQA_BUDGET_FILE", str(link))
    assert module.validated_budget_sha256() == expected


def test_budget_sha256_rejects_missing_file(tmp_path, monkeypatch):
    module = _load_generation_module()
    monkeypatch.setenv("QC_SQA_BUDGET_FILE", str(tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError):
        module.validated_budget_sha256()


def test_budget_sha256_requires_the_env_var(monkeypatch):
    module = _load_generation_module()
    monkeypatch.delenv("QC_SQA_BUDGET_FILE", raising=False)
    with pytest.raises(KeyError):
        module.validated_budget_sha256()


def _ring_states(variant, seed, n_states=5, n_atoms=9):
    pytest.importorskip("rdkit")
    pytest.importorskip("sklearn")
    variant_root = ROOT / f"example/molecular_design/{variant}"
    if str(variant_root) not in sys.path:
        sys.path.insert(0, str(variant_root))
    for name in [m for m in list(sys.modules) if m.startswith("qc_mol_design")]:
        del sys.modules[name]
    designer_mod = importlib.import_module("qc_mol_design.optimization.designer")
    qubo_mod = importlib.import_module("qc_mol_design.optimization.qubo")
    qubo = qubo_mod.MoleculeQUBO(["C"] * n_atoms, [4] * n_atoms)
    return designer_mod.MolecularDesigner._make_ring_initial_states(
        None, qubo, ["C"] * n_atoms, n_states=n_states, seed=seed
    )


@pytest.mark.parametrize("variant", ["sqa", "cim_sa"])
def test_ring_initial_states_reproducible_per_condition_seed(variant):
    first = _ring_states(variant, seed=1234)
    second = _ring_states(variant, seed=1234)
    assert first == second
    assert len(first) == 5


@pytest.mark.parametrize("variant", ["sqa", "cim_sa"])
def test_ring_initial_states_vary_with_condition_seed(variant):
    first = _ring_states(variant, seed=1234)
    second = _ring_states(variant, seed=999)
    assert first != second


@pytest.mark.parametrize("variant", ["sqa", "cim_sa"])
def test_ring_initial_states_ignore_global_random_stream(variant):
    """Warm-start sampling must not shift with unrelated global-stream use."""
    import random

    random.seed(20260614)
    seeded = _ring_states(variant, seed=1234)
    random.seed(7)  # unrelated consumption elsewhere must not matter
    again = _ring_states(variant, seed=1234)
    assert seeded == again
