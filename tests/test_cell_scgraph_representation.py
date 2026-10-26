"""scGraph must evaluate the requested embedding rather than relabel all of them."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(name="benchmark")
def fixture_benchmark():
    """Load the real optional evaluator and the example entry point."""
    pytest.importorskip("anndata")
    pytest.importorskip("scanpy")
    scgraph = pytest.importorskip("scgraph")
    if not hasattr(scgraph, "scGraph"):
        pytest.skip("The scgraph-eval distribution is required")
    path = Path(__file__).resolve().parents[1] / "example/qvae_cell/evaluate_benchmark.py"
    spec = importlib.util.spec_from_file_location("cell_scgraph_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, scgraph.scGraph


def make_cells(benchmark):
    """Two batches contain four cell types and distinct embedding geometries."""
    rng = np.random.default_rng(19)
    labels = np.tile(np.repeat(np.arange(4), 12), 2)
    rates = np.full((4, 24), 2.)
    for index in range(4):
        rates[index, index * 6:(index + 1) * 6] += 3 + index * 2
    cells = benchmark.anndata.AnnData(np.log1p(rng.poisson(rates[labels])).astype(np.float32))
    cells.obs["batch"] = pd.Categorical(["first"] * 48 + ["second"] * 48)
    cells.obs["cell_type"] = pd.Categorical([f"type_{label}" for label in labels])
    for key, centroids in (
        ("X_qvae", [[0, 0], [1, 0], [1, 4], [3, 2]]),
        ("X_other", [[0, 0], [4, 2], [0, 1], [1, 1]]),
    ):
        cells.obsm[key] = np.asarray(centroids)[labels] + rng.normal(0, .05, (96, 2))
    return cells


@pytest.mark.parametrize("rep_key, only_umap", [("X_qvae", True), ("X_other", False)])
def test_real_scgraph_scores_only_requested_representation(benchmark, tmp_path,
                                                          rep_key, only_umap):
    """Compare real backend metrics, final summary and CSV to a named-row oracle."""
    module, analyzer_class = benchmark
    cells = make_cells(module)
    original = cells.copy()
    reference_path = tmp_path / "reference.h5ad"
    cells.write_h5ad(reference_path)
    reference = analyzer_class(
        str(reference_path), "batch", "cell_type", only_umap=only_umap,
        trim_rate=.05, thres_batch=10, thres_celltype=5,
    ).main()
    reference_path.unlink()
    assert set(reference.index) == {"X_qvae", "X_other"}
    assert not np.allclose(reference.loc["X_qvae"], reference.loc["X_other"])
    assert np.isfinite(reference.to_numpy()).all()

    result = module.evaluate_scgraph(
        cells, rep_key, "batch", "cell_type", str(tmp_path), .05, 10, 5, only_umap,
    )

    assert len(result) == 1
    assert result["rep_key"].tolist() == [rep_key]
    assert result["metric_group"].tolist() == ["scgraph"]
    expected = reference.loc[[rep_key]].reset_index(drop=True)
    pd.testing.assert_frame_equal(result[expected.columns], expected)
    summary = module.make_long_summary([result], "cell_type")
    assert len(summary) == 3
    assert set(summary["rep_key"]) == {rep_key}
    assert set(summary["metric"]) == set(expected.columns)
    csv_path = tmp_path / "metrics.csv"
    module.save_result(result, str(csv_path), "scGraph metrics")
    pd.testing.assert_frame_equal(pd.read_csv(csv_path), result)
    assert not list(tmp_path.glob("*.h5ad"))
    np.testing.assert_array_equal(cells.X, original.X)
    pd.testing.assert_frame_equal(cells.obs, original.obs)
    assert set(cells.obsm) == set(original.obsm)
    for key in original.obsm:
        np.testing.assert_array_equal(cells.obsm[key], original.obsm[key])
