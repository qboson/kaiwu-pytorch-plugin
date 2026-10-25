"""Contract test for the ``evaluate_benchmark.main`` dispatch."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
EVALUATE = REPO / "example" / "qvae_cell" / "evaluate_benchmark.py"


class _FakeAdata:
    def __init__(self) -> None:
        self.obsm = {"X_qvae": np.zeros((6, 3))}
        self.obs = pd.DataFrame(
            {"final_annotation": ["A", "A", "B", "B", "C", "C"], "batch": [0, 0, 0, 1, 1, 1]}
        )
        self.n_obs = 6
        self.n_vars = 3
        self.uns: dict[str, object] = {}


def _load_benchmark():
    scanpy = types.ModuleType("scanpy")
    scanpy.pp = types.SimpleNamespace(
        pca=lambda *args, **kwargs: None,
        neighbors=lambda *args, **kwargs: None,
    )
    scanpy.tl = types.SimpleNamespace(
        diffmap=lambda *args, **kwargs: None,
        dpt=lambda *args, **kwargs: None,
    )
    anndata = types.ModuleType("anndata")
    previous = {name: sys.modules.get(name) for name in ("scanpy", "anndata")}
    sys.modules["scanpy"] = scanpy
    sys.modules["anndata"] = anndata
    try:
        spec = importlib.util.spec_from_file_location(
            "qvae_cell_evaluate_main_uut", EVALUATE
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _frame(name: str) -> pd.DataFrame:
    return pd.DataFrame({"metric": [name], "value": [1.0]})


def _install_stubs(module, adata):
    state: dict[str, object] = {"calls": [], "saved": [], "summaries": []}
    module.anndata.read_h5ad = lambda path: adata

    def recorder(name, value):
        def call(*args, **kwargs):
            state["calls"].append((name, args, kwargs))
            return value

        return call

    module.evaluate_clustering = recorder("clustering", _frame("clustering"))
    module.evaluate_classifier = recorder("classification", _frame("classification"))
    module.evaluate_scib = recorder("scib", _frame("scib"))
    module.evaluate_scgraph = recorder("scgraph", _frame("scgraph"))
    module.evaluate_dpt = recorder("dpt", _frame("dpt"))
    module.save_result = lambda df, path, label: state["saved"].append(
        (path, label)
    )
    module.make_long_summary = lambda results, label_key: (
        state["summaries"].append((len(results), label_key)) or "summary"
    )
    return state


def test_runs_only_the_selected_groups_and_saves_their_frames(tmp_path, monkeypatch):
    module = _load_benchmark()
    adata = _FakeAdata()
    state = _install_stubs(module, adata)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_benchmark.py",
            "--h5ad",
            str(tmp_path / "data.h5ad"),
            "--output-dir",
            str(tmp_path / "out"),
            "--metrics",
            "clustering,classification",
            "--resolutions",
            "0.2,0.4",
            "--n-neighbors",
            "9",
            "--classifier-test-size",
            "0.3",
            "--classifier-max-iter",
            "50",
        ],
    )

    module.main()

    names = [name for name, _, _ in state["calls"]]
    assert names == ["clustering", "classification"]
    clustering_args = state["calls"][0][1]
    assert clustering_args == (adata, "X_qvae", "final_annotation", [0.2, 0.4], 9, 42)
    classification_args = state["calls"][1][1]
    assert classification_args == (
        adata,
        "X_qvae",
        "final_annotation",
        0.3,
        42,
        50,
    )
    saved_paths = [path for path, _ in state["saved"]]
    assert saved_paths == [
        str(tmp_path / "out" / "X_qvae_clustering_metrics.csv"),
        str(tmp_path / "out" / "X_qvae_classification_metrics.csv"),
        str(tmp_path / "out" / "X_qvae_benchmark_summary.csv"),
    ]
    assert state["summaries"] == [(2, "final_annotation")]


def test_runs_classification_without_clustering(tmp_path, monkeypatch):
    module = _load_benchmark()
    adata = _FakeAdata()
    state = _install_stubs(module, adata)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_benchmark.py",
            "--h5ad",
            str(tmp_path / "data.h5ad"),
            "--output-dir",
            str(tmp_path / "out"),
            "--metrics",
            "classification",
        ],
    )

    module.main()

    assert [name for name, _, _ in state["calls"]] == ["classification"]
    assert [path for path, _ in state["saved"]] == [
        str(tmp_path / "out" / "X_qvae_classification_metrics.csv"),
        str(tmp_path / "out" / "X_qvae_benchmark_summary.csv"),
    ]


def test_dpt_requires_a_root_cell_type(tmp_path, monkeypatch):
    module = _load_benchmark()
    adata = _FakeAdata()
    state = _install_stubs(module, adata)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_benchmark.py",
            "--h5ad",
            str(tmp_path / "data.h5ad"),
            "--metrics",
            "dpt",
        ],
    )

    with pytest.raises(ValueError, match="--root-cell-type"):
        module.main()

    assert [name for name, _, _ in state["calls"]] == []
    assert state["saved"] == []


def test_optional_dependency_runtime_errors_are_skipped(tmp_path, monkeypatch, capsys):
    module = _load_benchmark()
    adata = _FakeAdata()
    state = _install_stubs(module, adata)

    def failing_scib(*args, **kwargs):
        raise RuntimeError("scib is unavailable")

    module.evaluate_scib = failing_scib
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_benchmark.py",
            "--h5ad",
            str(tmp_path / "data.h5ad"),
            "--metrics",
            "scib",
        ],
    )

    module.main()

    assert "Skipping scIB: scib is unavailable" in capsys.readouterr().out
    assert state["saved"] == []
    assert state["summaries"] == []
