"""Optional-dependency contract for the qvae_cell benchmark."""

import importlib.util
import os
import sys
import types

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for name in ("scanpy", "anndata"):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

_spec = importlib.util.spec_from_file_location(
    "qvae_cell_benchmark_optional",
    os.path.join(REPO_ROOT, "example", "qvae_cell", "evaluate_benchmark.py"),
)
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


def test_missing_scib_metrics_reports_a_runtime_error():
    """Without scib_metrics the group fails with the documented message."""
    with pytest.raises(RuntimeError, match="scib_metrics is not installed"):
        benchmark.evaluate_scib(None, "X_qvae", "batch", "label", 1)


def test_missing_scgraph_reports_a_runtime_error():
    """Without scgraph the group fails with the documented message."""
    with pytest.raises(RuntimeError, match="scgraph is not installed"):
        benchmark.evaluate_scgraph(
            None,
            "X_qvae",
            "batch",
            "label",
            ".",
            trim_rate=0.05,
            thres_batch=100,
            thres_celltype=10,
            only_umap=False,
        )
