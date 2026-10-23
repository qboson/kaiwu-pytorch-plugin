"""Clustering-agreement contract for the qvae_cell benchmark."""

import importlib.util
import os
import sys
import types

import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for name in ("scanpy", "anndata"):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

_spec = importlib.util.spec_from_file_location(
    "qvae_cell_benchmark_clustering",
    os.path.join(REPO_ROOT, "example", "qvae_cell", "evaluate_benchmark.py"),
)
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


def make_adata(cell_type, leiden):
    """Builds the minimal ``obs`` container the metric reads."""
    return types.SimpleNamespace(
        obs=pd.DataFrame({"cell_type": cell_type, "leiden": leiden})
    )


def test_perfect_agreement_scores_one():
    """Identical partitions yield perfect scores and the cluster count."""
    metrics = benchmark.compute_clustering_metrics(
        make_adata(["a", "a", "b", "b"], ["0", "0", "1", "1"]),
        "cell_type",
        "leiden",
    )
    assert metrics["ARI"] == pytest.approx(1.0)
    assert metrics["AMI"] == pytest.approx(1.0)
    assert metrics["NMI"] == pytest.approx(1.0)
    assert metrics["Homogeneity"] == pytest.approx(1.0)
    assert metrics["FMI"] == pytest.approx(1.0)
    assert metrics["n_clusters"] == 2


def test_cluster_count_uses_the_cluster_column():
    """``n_clusters`` counts distinct cluster values, not distinct labels."""
    metrics = benchmark.compute_clustering_metrics(
        make_adata(["a", "a", "b", "b"], ["0", "1", "2", "2"]),
        "cell_type",
        "leiden",
    )
    assert metrics["n_clusters"] == 3
    assert 0.0 <= metrics["ARI"] < 1.0
    assert set(metrics) == {"ARI", "AMI", "NMI", "Homogeneity", "FMI", "n_clusters"}
