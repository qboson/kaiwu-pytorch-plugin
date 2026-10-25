"""Contract test for ``example/qvae_cell/evaluate_benchmark.maybe_add_pca``."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVALUATE = REPO / "example" / "qvae_cell" / "evaluate_benchmark.py"


def _load_evaluate():
    scanpy = types.ModuleType("scanpy")
    scanpy.pp = types.SimpleNamespace(pca=lambda *args, **kwargs: None)
    anndata = types.ModuleType("anndata")
    previous = {name: sys.modules.get(name) for name in ("scanpy", "anndata")}
    sys.modules["scanpy"] = scanpy
    sys.modules["anndata"] = anndata
    try:
        spec = importlib.util.spec_from_file_location(
            "qvae_cell_evaluate_pca_uut", EVALUATE
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
    def __init__(self, n_obs: int, n_vars: int, obsm: dict | None = None) -> None:
        self.n_obs = n_obs
        self.n_vars = n_vars
        self.obsm = {} if obsm is None else obsm


def _record_calls(evaluate_module):
    calls: list[dict[str, object]] = []

    def fake_pca(adata, **kwargs):
        calls.append({"adata": adata, **kwargs})

    evaluate_module.sc.pp.pca = fake_pca
    return calls


def test_skips_when_pca_is_already_present():
    evaluate = _load_evaluate()
    calls = _record_calls(evaluate)
    adata = _Adata(100, 20, {"X_pca": "existing"})

    evaluate.maybe_add_pca(adata)

    assert calls == []
    assert adata.obsm == {"X_pca": "existing"}


def test_uses_the_smallest_of_the_three_bounds():
    evaluate = _load_evaluate()
    calls = _record_calls(evaluate)
    adata = _Adata(n_obs=100, n_vars=20)

    evaluate.maybe_add_pca(adata)

    # min(50, 100 - 1, 20 - 1) == 19
    assert calls == [{"adata": adata, "n_comps": 19}]


def test_caps_the_component_count_at_fifty():
    evaluate = _load_evaluate()
    calls = _record_calls(evaluate)
    adata = _Adata(n_obs=500, n_vars=400)

    evaluate.maybe_add_pca(adata)

    assert calls == [{"adata": adata, "n_comps": 50}]


def test_skips_when_one_component_is_not_enough():
    evaluate = _load_evaluate()
    calls = _record_calls(evaluate)

    evaluate.maybe_add_pca(_Adata(n_obs=100, n_vars=2))
    evaluate.maybe_add_pca(_Adata(n_obs=1, n_vars=100))

    assert calls == []


def test_runs_when_two_components_are_available():
    evaluate = _load_evaluate()
    calls = _record_calls(evaluate)
    adata = _Adata(n_obs=100, n_vars=3)

    evaluate.maybe_add_pca(adata)

    assert calls == [{"adata": adata, "n_comps": 2}]
