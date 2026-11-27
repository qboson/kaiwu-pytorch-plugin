"""Replay picks the currently configured frozen manifest in either solver arm."""

import importlib
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture(params=["sqa", "cim_sa"])
def replay_module(request, monkeypatch):
    root = Path(__file__).resolve().parents[1] / "example" / "molecular_design"
    monkeypatch.syspath_prepend(str(root / request.param))
    package = "qc_mol_design"
    original = {
        key: sys.modules.pop(key)
        for key in list(sys.modules)
        if key == package or key.startswith(f"{package}.")
    }
    try:
        yield importlib.import_module(f"{package}.optimization.history_prefix")
    finally:
        for key in list(sys.modules):
            if key == package or key.startswith(f"{package}."):
                sys.modules.pop(key)
        sys.modules.update(original)


def create_manifest(directory, bitstring):
    directory.mkdir()
    row = {
        "global_step": 0, "A_bitstring": bitstring, "valid": True,
        "target_hit": False, "solver_seed": 12, "solver_returned_reads": 50,
    }
    path = directory / "prefix.json"
    path.write_text(json.dumps({"prefix_steps": 1, "rows": [row]}), encoding="utf-8")
    return row


def test_replay_tracks_relative_paths_and_disabling_after_a_load(
    replay_module, monkeypatch, tmp_path
):
    first = tmp_path / "实验甲"
    second = tmp_path / "实验乙"
    row_a = create_manifest(first, "001")
    row_b = create_manifest(second, "110")
    monkeypatch.setenv("QC_HISTORY_PREFIX_FILE", "prefix.json")

    monkeypatch.chdir(first)
    replay_module.validate_selected(row_a)
    monkeypatch.chdir(second)
    replay_module.validate_selected(row_b)
    with pytest.raises(RuntimeError, match="Historical selection changed"):
        replay_module.validate_selected(row_a)

    monkeypatch.delenv("QC_HISTORY_PREFIX_FILE")
    assert replay_module.history() is None
    replay_module.validate_selected(row_a)


def test_starting_with_replay_disabled_does_not_cache_none(replay_module, monkeypatch, tmp_path):
    monkeypatch.delenv("QC_HISTORY_PREFIX_FILE", raising=False)
    assert replay_module.history() is None
    row = create_manifest(tmp_path / "active", "001")
    monkeypatch.setenv("QC_HISTORY_PREFIX_FILE", str(tmp_path / "active" / "prefix.json"))

    with pytest.raises(RuntimeError, match="Historical selection changed"):
        replay_module.validate_selected({**row, "A_bitstring": "100"})
