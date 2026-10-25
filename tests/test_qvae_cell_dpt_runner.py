"""Contract test for ``example/qvae_cell/evaluate_benchmark.run_dpt``."""

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


def _load_evaluate():
    scanpy = types.ModuleType("scanpy")
    scanpy.pp = types.SimpleNamespace(neighbors=lambda *args, **kwargs: None)
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
            "qvae_cell_evaluate_dpt_uut", EVALUATE
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
    def __init__(self, labels: list[str]) -> None:
        names = [f"cell{index}" for index in range(len(labels))]
        self.obs = pd.DataFrame({"cell_type": labels}, index=names)
        self.obs_names = self.obs.index
        self.uns: dict[str, object] = {}


def _install_stubs(evaluate):
    events: list[tuple[str, object]] = []

    def fake_neighbors(adata, **kwargs):
        events.append(("neighbors", kwargs))

    def fake_diffmap(adata):
        events.append(("diffmap", None))

    def fake_dpt(adata):
        events.append(("dpt", None))
        adata.obs["dpt_pseudotime"] = np.linspace(0.0, 1.0, len(adata.obs))

    evaluate.sc.pp.neighbors = fake_neighbors
    evaluate.sc.tl.diffmap = fake_diffmap
    evaluate.sc.tl.dpt = fake_dpt
    return events


def test_runs_the_pipeline_and_writes_the_requested_column():
    evaluate = _load_evaluate()
    events = _install_stubs(evaluate)
    adata = _Adata(["A", "B", "B", "C"])

    result = evaluate.run_dpt(adata, "X_qvae", "B", "cell_type", 15, "pt_qvae")

    assert events[0] == ("neighbors", {"use_rep": "X_qvae", "n_neighbors": 15})
    assert [name for name, _ in events[1:]] == ["diffmap", "dpt"]
    # The root is the first "B" cell, which sits at position 1 here.
    assert adata.uns["iroot"] == 1
    assert "pt_qvae" in adata.obs
    assert result.equals(adata.obs["pt_qvae"])
    assert np.allclose(result.to_numpy(), np.linspace(0.0, 1.0, 4))


def test_rejects_an_unknown_root_cell_type_before_diffusion():
    evaluate = _load_evaluate()
    events = _install_stubs(evaluate)
    adata = _Adata(["A", "B"])

    with pytest.raises(ValueError, match="not found"):
        evaluate.run_dpt(adata, "X_qvae", "Z", "cell_type", 10, "pt_qvae")

    assert [name for name, _ in events] == ["neighbors"]
    assert "pt_qvae" not in adata.obs
    assert "iroot" not in adata.uns


def test_uses_the_first_cell_of_the_root_group_as_its_start():
    evaluate = _load_evaluate()
    _install_stubs(evaluate)
    adata = _Adata(["B", "B", "A"])

    evaluate.run_dpt(adata, "X_qvae", "B", "cell_type", 5, "pt_qvae")

    # Both "B" cells are valid roots; the first one wins.
    assert adata.uns["iroot"] == 0
