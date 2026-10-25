"""DPT evaluation contract for the qvae_cell benchmark."""

import importlib.util
import os
import sys
import types

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for name in ("scanpy", "anndata"):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

_spec = importlib.util.spec_from_file_location(
    "qvae_cell_benchmark_dpt",
    os.path.join(REPO_ROOT, "example", "qvae_cell", "evaluate_benchmark.py"),
)
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


class FakeAdata:
    """Minimal AnnData stand-in supporting copy/obs_names/obsm."""

    def __init__(self, with_pca):
        self.obs_names = pd.Index(["c1", "c2", "c3"])
        self.obs = pd.DataFrame({"cell_type": ["root", "b", "a"]})
        self.obsm = {"X_qvae": np.zeros((3, 2))}
        if with_pca:
            self.obsm["X_pca"] = np.zeros((3, 2))

    def copy(self):
        return FakeAdata("X_pca" in self.obsm)


def test_pseudotime_count_and_trajectory_agreement_are_reported(monkeypatch):
    """Non-NA cells are counted and the anti-correlated baseline scores zero."""
    def fake_run_dpt(adata, rep_key, root_cell_type, label_key, n_neighbors, out_key):
        if rep_key == "X_pca":
            return pd.Series([3.0, 2.0, 1.0], index=adata.obs_names)
        return pd.Series([1.0, np.nan, 3.0], index=adata.obs_names)

    monkeypatch.setattr(benchmark, "run_dpt", fake_run_dpt)
    table = benchmark.evaluate_dpt(
        FakeAdata(with_pca=True),
        "X_qvae",
        "cell_type",
        "root",
        n_neighbors=5,
    )
    assert len(table) == 1
    row = table.iloc[0]
    assert row["metric_group"] == "dpt"
    assert row["n_cells_with_dpt"] == 2
    assert row["kendall_tau_vs_X_pca"] == -1.0
    assert row["trajectory_conservation_vs_X_pca"] == 0.0


def test_without_a_pca_baseline_no_trajectory_columns_are_added(monkeypatch):
    """The optional comparison columns only appear when X_pca exists."""
    def fake_run_dpt(adata, rep_key, root_cell_type, label_key, n_neighbors, out_key):
        return pd.Series([1.0, 2.0, 3.0], index=adata.obs_names)

    monkeypatch.setattr(benchmark, "run_dpt", fake_run_dpt)
    table = benchmark.evaluate_dpt(
        FakeAdata(with_pca=False), "X_qvae", "cell_type", "root", n_neighbors=5
    )
    row = table.iloc[0]
    assert row["n_cells_with_dpt"] == 3
    assert "kendall_tau_vs_X_pca" not in table.columns
