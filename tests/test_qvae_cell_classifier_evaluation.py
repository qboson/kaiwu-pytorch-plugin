"""Contract test for ``evaluate_benchmark.evaluate_classifier``."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EVALUATE = REPO / "example" / "qvae_cell" / "evaluate_benchmark.py"


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
            "qvae_cell_evaluate_classifier_uut", EVALUATE
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


class _Adata:
    def __init__(self, embedding, labels):
        self.obsm = {"X_qvae": embedding}
        self.obs = pd.DataFrame({"cell_type": labels})


def _separable_adata():
    rng = np.random.default_rng(0)
    low = rng.normal(loc=-10.0, scale=0.1, size=(20, 2))
    high = rng.normal(loc=10.0, scale=0.1, size=(20, 2))
    embedding = np.vstack([low, high])
    labels = ["type_a"] * 20 + ["type_b"] * 20
    return _Adata(embedding, labels)


def test_reports_perfect_scores_for_separable_embeddings():
    benchmark = _load_benchmark()

    frame = benchmark.evaluate_classifier(
        _separable_adata(), "X_qvae", "cell_type", 0.25, 7, 200
    )

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["metric_group"] == "classification"
    assert row["rep_key"] == "X_qvae"
    assert row["label_key"] == "cell_type"
    assert row["classifier"] == "logistic_regression"
    assert row["test_size"] == 0.25
    assert row["n_classes"] == 2
    assert row["accuracy"] == 1.0
    assert row["precision_weighted"] == 1.0
    assert row["recall_weighted"] == 1.0
    assert row["f1_weighted"] == 1.0
