"""A named DPT reference is reproducible across embeddings and row order."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def benchmark(monkeypatch):
    anndata = pytest.importorskip("anndata", exc_type=ImportError)
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    pytest.importorskip("sklearn", exc_type=ImportError)
    neighbors_calls, roots = [], []
    scanpy = ModuleType("scanpy")

    def pseudotime(adata):
        position = adata.uns["iroot"]
        roots.append(adata.obs_names[position])
        adata.obs["dpt_pseudotime"] = np.abs(np.arange(adata.n_obs) - position)

    scanpy.pp = SimpleNamespace(neighbors=lambda obj, **opts: neighbors_calls.append(opts["use_rep"]))
    scanpy.tl = SimpleNamespace(diffmap=lambda obj: None, dpt=pseudotime)
    monkeypatch.setitem(sys.modules, "scanpy", scanpy)
    file = Path(__file__).resolve().parents[1] / "example" / "qvae_cell" / "evaluate_benchmark.py"
    spec = importlib.util.spec_from_file_location("kpp_dpt_benchmark", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observations = pd.DataFrame(
        {"label": ["start", "end", "start"]}, index=["cell-a", "cell-b", "cell-c"]
    )
    matrix = np.ones((3, 2))
    data = anndata.AnnData(matrix, obs=observations, obsm={"X_qvae": matrix, "X_pca": matrix})
    return module, data, neighbors_calls, roots


def test_exact_root_is_shared_and_reported_in_cli(benchmark, monkeypatch, tmp_path):
    module, data, calls, roots = benchmark
    reordered = data[[2, 0, 1]].copy()
    monkeypatch.setattr(module.anndata, "read_h5ad", lambda path: reordered)
    monkeypatch.setattr(sys, "argv", [
        "evaluate_benchmark.py", "--h5ad", "example.h5ad", "--label-key", "label",
        "--metrics", "dpt", "--root-cell", "cell-c", "--output-dir", str(tmp_path),
    ])
    module.main()
    result = module.pd.read_csv(tmp_path / "X_qvae_dpt_metrics.csv")
    assert calls == ["X_qvae", "X_pca"]
    assert roots == ["cell-c", "cell-c"]
    assert result["root_cell"].item() == "cell-c"
    assert result["root_cell_type"].item() == "start"
    default = module.evaluate_dpt(data, "X_qvae", "label", "start", 2)
    assert default["root_cell"].item() == "cell-a"


@pytest.mark.parametrize("name, label", [("missing", None), ("cell-c", "end")])
def test_invalid_named_root_is_rejected_before_graph_build(benchmark, name, label):
    module, data, calls, _ = benchmark
    with pytest.raises(ValueError, match="root"):
        module.evaluate_dpt(data, "X_qvae", "label", label, 2, root_cell=name)
    assert calls == []
