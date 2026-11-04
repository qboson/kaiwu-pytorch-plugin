"""Regression tests for the example-layer security follow-up round.

Pins three behaviours introduced after the first hardening pass:
  * every torch.load call site restricts unpickling to weights-only
    checkpoints, including the format-v3 molecular payload shape;
  * file-hash entry points canonicalise their path and reject missing
    files instead of hashing whatever the caller happens to pass;
  * the remaining example samplers (built-in Metropolis SA, QUBO
    perturbation noise, t-SNE sub-sampling inputs, synthetic datasets,
    dataset splitting) draw from seeded generators without touching the
    global random streams.
"""

import hashlib
import importlib.util
import os
import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
MOLECULAR_ROOT = ROOT / "example/molecular_design"


def _load_molecular_variant(variant, relpath, name):
    variant_root = MOLECULAR_ROOT / variant
    if str(variant_root) not in sys.path:
        sys.path.insert(0, str(variant_root))
    for stale in [m for m in list(sys.modules) if m.startswith("qc_mol_design")]:
        del sys.modules[stale]
    return importlib.import_module(f"qc_mol_design.{relpath}")


def _run_generation_module(variant):
    pytest.importorskip("pandas")
    script_root = MOLECULAR_ROOT / variant
    if str(script_root) not in sys.path:
        sys.path.insert(0, str(script_root))
    for stale in [m for m in list(sys.modules) if m.startswith("qc_mol_design")]:
        del sys.modules[stale]
    script = script_root / "run_generation_strict.py"
    spec = importlib.util.spec_from_file_location(f"run_generation_strict_{variant}", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── torch.load weight-only unpickling ────────────────────────────────────────

def test_molecular_v3_checkpoint_roundtrip_under_weights_only(tmp_path):
    """The documented format-v3 payload (two state_dicts plus primitive
    provenance) must keep loading under the now-mandatory weights_only=True."""
    from collections import OrderedDict

    torch = pytest.importorskip("torch")
    checkpoint = {
        "gcn": OrderedDict([("layer.weight", torch.randn(4, 4))]),
        "crbm": OrderedDict([("w", torch.randn(3, 3))]),
        "format_version": 3,
        "feature_schema": "paper_nine_heavy_atom_onehot_v1",
    }
    path = tmp_path / "checkpoint_qc.pt"
    torch.save(checkpoint, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["format_version"] == 3
    assert loaded["feature_schema"] == "paper_nine_heavy_atom_onehot_v1"
    assert set(loaded["gcn"]) == {"layer.weight"}


# ── file-hash entry points canonicalise and validate ─────────────────────────

@pytest.mark.parametrize("variant", ["sqa", "cim_sa"])
def test_manifest_sha256_matches_hashlib(variant):
    data_mod = _load_molecular_variant(variant, "data", "data")
    probe = MOLECULAR_ROOT / variant / "qc_mol_design" / "config.py"
    expected = hashlib.sha256(probe.read_bytes()).hexdigest()
    assert data_mod.manifest_sha256(probe) == expected
    assert data_mod.manifest_sha256(str(probe)) == expected


@pytest.mark.parametrize("variant", ["sqa", "cim_sa"])
def test_manifest_sha256_rejects_missing_file(tmp_path, variant):
    data_mod = _load_molecular_variant(variant, "data", "data")
    with pytest.raises(FileNotFoundError):
        data_mod.manifest_sha256(tmp_path / "absent_manifest.json")


def test_matched_script_sha_rejects_missing_file(tmp_path):
    script = MOLECULAR_ROOT / "scripts" / "run_sqa_cim_matched.py"
    spec = importlib.util.spec_from_file_location("run_sqa_cim_matched", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(FileNotFoundError):
        module.sha(tmp_path / "absent_input.csv")


# ── seeded built-in Metropolis SA ────────────────────────────────────────────

def _builtin_sa(variant, *args, **kwargs):
    qubo_mod = _load_molecular_variant(variant, "optimization.qubo", "qubo")
    return qubo_mod._solve_builtin_sa(*args, **kwargs)


@pytest.mark.parametrize("variant", ["sqa", "cim_sa"])
def test_builtin_sa_same_seed_same_result(variant):
    import numpy as np

    q = np.array([[4.0, -2.0, 0.0], [0.0, 4.0, -2.0], [0.0, 0.0, 6.0]])
    first = _builtin_sa(variant, q, num_reads=3, num_sweeps=200, seed=7)
    second = _builtin_sa(variant, q, num_reads=3, num_sweeps=200, seed=7)
    assert first == second


@pytest.mark.parametrize("variant", ["sqa", "cim_sa"])
def test_builtin_sa_does_not_touch_global_streams(variant):
    import numpy as np

    q = np.array([[4.0, -2.0, 0.0], [0.0, 4.0, -2.0], [0.0, 0.0, 6.0]])
    random.seed(12345)
    np.random.seed(12345)
    py_state = random.getstate()
    np_state = np.random.get_state()
    _builtin_sa(variant, q, num_reads=3, num_sweeps=200, seed=7)
    _builtin_sa(
        variant, q, num_reads=2, num_sweeps=50, seed=7,
        initial_states=[{0: 1, 1: 0, 2: 1}],
    )
    assert random.getstate() == py_state
    fresh = np.random.get_state()
    assert fresh[0] == np_state[0]
    assert np.array_equal(fresh[1], np_state[1])
    assert fresh[2:] == np_state[2:]


@pytest.mark.parametrize("variant", ["sqa", "cim_sa"])
def test_builtin_sa_seed_changes_single_sweep_sampling(variant):
    """With one sweep the outcome still depends on the supplied seed."""
    import numpy as np

    q = np.array([[4.0, -2.0, 3.0], [0.0, 4.0, -1.0], [0.0, 0.0, 5.0]])
    a = _builtin_sa(variant, q, num_reads=3, num_sweeps=1, seed=7)
    b = _builtin_sa(variant, q, num_reads=3, num_sweeps=1, seed=8)
    assert a != b


@pytest.mark.parametrize("variant", ["sqa", "cim_sa"])
def test_builtin_sa_warm_start_path_runs(variant):
    import numpy as np

    q = np.array([[4.0, -2.0, 0.0], [0.0, 4.0, -2.0], [0.0, 0.0, 6.0]])
    solutions = _builtin_sa(
        variant, q, num_reads=2, num_sweeps=50, seed=7,
        initial_states=[{0: 1, 1: 0, 2: 1}],
    )
    assert len(solutions) == 2


# ── seeded QUBO perturbation noise ───────────────────────────────────────────

@pytest.mark.parametrize("variant", ["sqa", "cim_sa"])
def test_qubo_build_perturbation_deterministic_per_seed(variant):
    import numpy as np

    qubo_mod = _load_molecular_variant(variant, "optimization.qubo", "qubo")
    qubo = qubo_mod.MoleculeQUBO(["C"] * 4, [4] * 4)
    first = qubo.build(None, 1.0, perturbation=0.5, seed=11)
    second = qubo.build(None, 1.0, perturbation=0.5, seed=11)
    other = qubo.build(None, 1.0, perturbation=0.5, seed=12)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


@pytest.mark.parametrize("variant", ["sqa", "cim_sa"])
def test_qubo_build_perturbation_does_not_touch_global_streams(variant):
    import numpy as np

    qubo_mod = _load_molecular_variant(variant, "optimization.qubo", "qubo")
    qubo = qubo_mod.MoleculeQUBO(["C"] * 4, [4] * 4)
    random.seed(5)
    np.random.seed(5)
    py_state = random.getstate()
    np_state = np.random.get_state()
    qubo.build(None, 1.0, perturbation=0.5, seed=11)
    assert random.getstate() == py_state
    fresh = np.random.get_state()
    assert np.array_equal(fresh[1], np_state[1])
    assert fresh[2:] == np_state[2:]


# ── deterministic synthetic feature-selection datasets ───────────────────────

def _feature_datasets_module():
    pkg_root = ROOT / "example" / "feature_selection"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    for stale in [m for m in list(sys.modules)
                  if m == "feature_selection_datasets"]:
        del sys.modules[stale]
    return importlib.import_module("feature_selection_datasets")


def test_feature_selection_datasets_are_reproducible():
    torch = pytest.importorskip("torch")
    module = _feature_datasets_module()
    first = module.build_linear_regression_dataset()
    second = module.build_linear_regression_dataset()
    assert torch.equal(first.inputs, second.inputs)
    assert torch.equal(first.targets, second.targets)
    assert first.signal_features == second.signal_features
    cnn_a = module.build_cnn_dataset()
    cnn_b = module.build_cnn_dataset()
    assert torch.equal(cnn_a.inputs, cnn_b.inputs)
    seq_a = module.build_sequence_dataset()
    seq_b = module.build_sequence_dataset()
    assert torch.equal(seq_a.inputs, seq_b.inputs)


# ── deterministic dataset splitting helper ───────────────────────────────────

def test_qdiffusion_split_is_deterministic_per_seed(monkeypatch):
    """split_train_val_test shuffles with a seeded generator, so the same
    seed yields the same split regardless of surrounding global state."""
    workflows_dir = ROOT / "example" / "qdiffusion" / "dplm" / "workflows"

    stub_pkg = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("utils", None)
    )
    stub_pkg.__path__ = []
    sys.modules["utils"] = stub_pkg
    for sub, body in (
        ("utils.io", "def normalize_decoded_sequence(s):\n    return s\n"
         "def save_markdown(*a, **k):\n    pass\n"
         "def write_fasta_records(*a, **k):\n    pass\n"
         "def write_tsv_rows(*a, **k):\n    pass\n"),
        ("utils.metrics", "class QualitySummary:\n    pass\n"),
        ("utils.runtime", "def encode_sequence(*a, **k):\n    pass\n"
         "def seed_torch(*a, **k):\n    pass\n"),
    ):
        stub = importlib.util.module_from_spec(
            importlib.util.spec_from_loader(sub, None)
        )
        exec(body, stub.__dict__)
        sys.modules[sub] = stub
    monkeypatch.syspath_prepend(str(workflows_dir))
    try:
        helpers = importlib.import_module("workflow_helpers")
        records = [(f"header_{i}", "ACGT") for i in range(10)]
        first = helpers.split_train_val_test(
            records, val_ratio=0.2, test_ratio=0.2, seed=42
        )
        second = helpers.split_train_val_test(
            records, val_ratio=0.2, test_ratio=0.2, seed=42
        )
        assert first == second
        assert [h for h, _ in first[0]] != [
            h for h, _ in helpers.split_train_val_test(
                records, val_ratio=0.2, test_ratio=0.2, seed=43
            )[0]
        ]
        merged = [h for h, _ in first[0] + first[1] + first[2]]
        assert sorted(merged) == sorted(h for h, _ in records)
    finally:
        sys.path.remove(str(workflows_dir))
        for name in ("workflow_helpers", "utils", "utils.io",
                     "utils.metrics", "utils.runtime"):
            sys.modules.pop(name, None)


# ── weights_only across the example-layer call sites ────────────────────────

def test_example_torch_load_sites_pass_weights_only():
    """Static pin: every torch.load in example/ states weights_only."""
    flagged = []
    for path in (ROOT / "example").rglob("*.py"):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for index, line in enumerate(lines):
            if "torch.load(" not in line:
                continue
            window = lines[index:index + 5]
            if not any("weights_only" in extra for extra in window):
                flagged.append(f"{path.relative_to(ROOT)}:{index + 1}")
    assert flagged == []
