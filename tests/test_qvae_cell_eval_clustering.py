"""Leiden-clustering evaluation contract for the qvae_cell benchmark."""

import importlib.util
import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for name in ("scanpy", "anndata"):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

_spec = importlib.util.spec_from_file_location(
    "qvae_cell_benchmark_clustering_eval",
    os.path.join(REPO_ROOT, "example", "qvae_cell", "evaluate_benchmark.py"),
)
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


class FakeScanpy:
    """Assigns deterministic clusters per resolution and records calls."""

    def __init__(self):
        self.calls = []
        self.pp = SimpleNamespace(neighbors=self.neighbors)
        self.tl = SimpleNamespace(leiden=self.leiden)

    def neighbors(self, adata, **kwargs):
        self.calls.append(("neighbors", kwargs))

    def leiden(self, adata, **kwargs):
        self.calls.append(("leiden", kwargs))
        labels = adata.obs["cell_type"].tolist()
        resolution = kwargs["resolution"]
        if resolution == 0.2:
            values = ["0"] * len(labels)
        elif resolution == 0.5:
            values = list(labels)
        else:
            values = ["merged" if label == "0" else "other" for label in labels]
        adata.obs[kwargs["key_added"]] = values


def build_adata():
    """Six cells with three labels and one embedding."""
    return SimpleNamespace(
        obs=pd.DataFrame({"cell_type": ["0", "0", "1", "1", "2", "2"]}),
        obsm={"X_qvae": np.zeros((6, 2))},
    )


def test_neighbors_are_built_once_and_each_resolution_is_scored_and_sorted():
    """One neighbor graph is reused; rows are sorted by decreasing ARI."""
    fake = FakeScanpy()
    original = benchmark.sc
    benchmark.sc = fake
    try:
        table = benchmark.evaluate_clustering(
            build_adata(),
            "X_qvae",
            "cell_type",
            resolutions=[0.2, 0.5, 0.8],
            n_neighbors=10,
            random_state=7,
        )
    finally:
        benchmark.sc = original

    neighbor_calls = [kwargs for kind, kwargs in fake.calls if kind == "neighbors"]
    assert neighbor_calls == [
        {"use_rep": "X_qvae", "n_neighbors": 10, "key_added": "X_qvae_neighbors"}
    ]
    leiden_calls = [kwargs for kind, kwargs in fake.calls if kind == "leiden"]
    assert [call["resolution"] for call in leiden_calls] == [0.2, 0.5, 0.8]
    assert all(call["neighbors_key"] == "X_qvae_neighbors" for call in leiden_calls)
    assert all(call["random_state"] == 7 for call in leiden_calls)

    assert len(table) == 3
    assert list(table["ARI"]) == sorted(table["ARI"], reverse=True)
    assert table.iloc[0]["resolution"] == 0.5
    assert set(table["metric_group"]) == {"clustering"}
    assert set(table["rep_key"]) == {"X_qvae"}
    assert {"ARI", "AMI", "NMI", "Homogeneity", "FMI", "n_clusters"} <= set(table.columns)
