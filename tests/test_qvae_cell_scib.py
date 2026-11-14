"""Optional scIB integration regressions for the cell benchmark example."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata")
pytest.importorskip("scanpy")
scib_metrics = pytest.importorskip("scib_metrics")

from scib_metrics import benchmark as scib_benchmark
from scib_metrics.benchmark import _core


@pytest.fixture
def evaluator():
    path = (
        Path(__file__).resolve().parents[1]
        / "example/qvae_cell/evaluate_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("qvae_cell_evaluate_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "rep_key,has_pca", [("X_pca", True), ("X_qvae", True), ("X_qvae", False)]
)
def test_scib_preserves_embedding_and_later_dpt(evaluator, monkeypatch, rep_key, has_pca):
    rng = np.random.default_rng(12)
    adata = anndata.AnnData(
        rng.normal(size=(12, 6)),
        obs=pd.DataFrame(
            {"batch": ["a", "b"] * 6, "cell_type": ["root"] * 6 + ["leaf"] * 6},
            index=[f"cell-{i}" for i in range(12)],
        ),
    )
    adata.obsm["X_qvae"] = rng.normal(size=(12, 2))
    if has_pca:
        # Deliberately distinct from PCA recomputed from X.
        adata.obsm["X_pca"] = 100 + rng.normal(size=(12, 3))
        adata.varm["PCs"] = rng.normal(size=(6, 3))
        adata.uns["pca"] = {"variance": np.array([3.0, 2.0, 1.0])}
    original = adata.copy()

    # Keep real scIB preparation, metric dispatch and result aggregation; isolate
    # only expensive neighbor/metric calculations from this input-data regression.
    bio_conservation = scib_benchmark.BioConservation
    batch_correction = scib_benchmark.BatchCorrection
    monkeypatch.setattr(
        scib_benchmark,
        "BioConservation",
        lambda: bio_conservation(
            isolated_labels=False, nmi_ari_cluster_labels_kmeans=False,
            silhouette_label=True, clisi_knn=False,
        ),
    )
    monkeypatch.setattr(
        scib_benchmark,
        "BatchCorrection",
        lambda: batch_correction(
            bras=False, ilisi_knn=False, kbet_per_label=False,
            graph_connectivity=False, pcr_comparison=True,
        ),
    )
    monkeypatch.setattr(
        _core, "pynndescent",
        lambda *args, **kwargs: SimpleNamespace(subset_neighbors=lambda n: None),
    )
    observed = {}

    def silhouette(embedding, labels):
        observed["embedding"] = embedding.copy()
        return 0.75

    def pcr(baseline, embedding, batches, categorical):
        observed["baseline"] = baseline.copy()
        assert categorical is True
        return 0.5

    monkeypatch.setattr(scib_metrics, "silhouette_label", silhouette)
    monkeypatch.setattr(scib_metrics, "pcr_comparison", pcr)
    monkeypatch.setattr(
        evaluator, "run_dpt",
        lambda data, key, *args: pd.Series(data.obsm[key][:, 0], index=data.obs_names),
    )
    dpt_before = evaluator.evaluate_dpt(adata, rep_key, "cell_type", "root", 5)

    result = evaluator.evaluate_scib(adata, rep_key, "batch", "cell_type", n_jobs=1)

    np.testing.assert_array_equal(observed["embedding"], original.obsm[rep_key])
    if has_pca:
        np.testing.assert_array_equal(observed["baseline"], original.obsm["X_pca"])
    else:
        assert observed["baseline"].shape == (12, 5)
        assert np.isfinite(observed["baseline"]).all()
    np.testing.assert_array_equal(adata.X, original.X)
    pd.testing.assert_frame_equal(adata.obs, original.obs)
    pd.testing.assert_frame_equal(adata.var, original.var)
    for mapping in ("obsm", "varm"):
        assert set(getattr(adata, mapping)) == set(getattr(original, mapping))
        for key in getattr(original, mapping):
            np.testing.assert_array_equal(
                getattr(adata, mapping)[key], getattr(original, mapping)[key]
            )
    np.testing.assert_equal(adata.uns, original.uns)
    pd.testing.assert_frame_equal(
        evaluator.evaluate_dpt(adata, rep_key, "cell_type", "root", 5), dpt_before
    )
    assert list(result.index) == [0]
    assert result.loc[0, "metric_group"] == "scib"
    assert result.loc[0, "rep_key"] == rep_key
    assert result.loc[0, "Silhouette label"] == pytest.approx(0.75)
    assert result.loc[0, "PCR comparison"] == pytest.approx(0.5)
    assert result.loc[0, "Total"] == pytest.approx(0.65)
