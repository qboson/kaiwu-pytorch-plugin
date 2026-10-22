"""Regression check for rare labels in the single-cell benchmark."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd


def test_singleton_label_stays_available_for_classifier_training(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "qvae_cell_benchmark", Path(__file__).with_name("evaluate_benchmark.py")
    )
    benchmark = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "anndata", ModuleType("anndata"))
    monkeypatch.setitem(sys.modules, "scanpy", ModuleType("scanpy"))
    spec.loader.exec_module(benchmark)

    # With the original unstratified split, seed 2 puts the rare class only
    # in the test set and LogisticRegression.fit sees a single class.
    adata = SimpleNamespace(
        obsm={"X_qvae": np.arange(10, dtype=float).reshape(5, 2)},
        obs=pd.DataFrame({"cell_type": ["common"] * 4 + ["rare"]}),
    )
    result = benchmark.evaluate_classifier(adata, "X_qvae", "cell_type", 0.4, 2, 100)

    assert result.loc[0, "n_classes"] == 2
    assert 0 <= result.loc[0, "accuracy"] <= 1
