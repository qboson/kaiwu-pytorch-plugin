"""Contract test for evaluate_classifier with a single class."""

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
            "qvae_cell_classifier_single_uut", EVALUATE
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
    def __init__(self) -> None:
        self.obsm = {"X_qvae": np.zeros((8, 2), dtype=np.float32)}
        self.obs = pd.DataFrame({"cell_type": ["A"] * 8})


def test_rejects_a_single_class_cohort():
    benchmark = _load_benchmark()

    with pytest.raises(ValueError, match="at least two classes"):
        benchmark.evaluate_classifier(_Adata(), "X_qvae", "cell_type", 0.25, 0, 100)
