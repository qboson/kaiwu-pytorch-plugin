"""Long-form summary contract for the qvae_cell benchmark."""

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
    "qvae_cell_benchmark_long", os.path.join(REPO_ROOT, "example", "qvae_cell", "evaluate_benchmark.py")
)
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


def test_metric_columns_are_melted_and_the_label_column_is_filled():
    """Per-task widetables become one metric/value row per non-null metric."""
    classification = pd.DataFrame(
        {"metric_group": ["classification"], "rep_key": ["X_qvae"], "accuracy": [0.9]}
    )
    clustering = pd.DataFrame(
        {
            "metric_group": ["clustering"],
            "rep_key": ["X_qvae"],
            "label_key": [np.nan],
            "resolution": [0.5],
            "ARI": [0.8],
        }
    )
    summary = benchmark.make_long_summary([classification, clustering], "cell_type")

    assert set(summary.columns) >= {
        "task",
        "rep_key",
        "label_key",
        "resolution",
        "classifier",
        "metric",
        "value",
    }
    assert set(summary["task"]) == {"classification", "clustering"}
    assert set(summary["label_key"]) == {"cell_type"}
    assert set(summary["metric"]) >= {"accuracy", "ARI"}
    assert summary["value"].notna().all()
    accuracy_row = summary[summary["metric"] == "accuracy"].iloc[0]
    assert accuracy_row["value"] == 0.9
