"""Missing annotations are not biological classes in clustering metrics."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from sklearn import metrics


@pytest.fixture
def benchmark():
    path = Path(__file__).resolve().parents[1] / "example/qvae_cell/evaluate_benchmark.py"
    spec = importlib.util.spec_from_file_location("cell_missing_annotation_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pairwise_missing_annotations_are_excluded(benchmark):
    obs = pd.DataFrame({"label": ["A", "A", "B", "B", None, "A", "nan"],
                        "cluster": ["0", "1", "1", "1", "unknown", None, "2"]})
    adata = AnnData(np.zeros((len(obs), 2)), obs=obs)
    expected_labels = ["A", "A", "B", "B", "nan"]
    expected_clusters = ["0", "1", "1", "1", "2"]
    result = benchmark.compute_clustering_metrics(adata, "label", "cluster")
    for key, scorer in [("ARI", metrics.adjusted_rand_score), ("AMI", metrics.adjusted_mutual_info_score),
                        ("NMI", metrics.normalized_mutual_info_score), ("Homogeneity", metrics.homogeneity_score),
                        ("FMI", metrics.fowlkes_mallows_score)]:
        assert result[key] == pytest.approx(scorer(expected_labels, expected_clusters))
    assert result["n_clusters"] == 3
    assert result["n_evaluated_cells"] == 5
    assert adata.n_obs == 7


def test_categorical_missing_values_are_excluded(benchmark):
    obs = pd.DataFrame({"label": pd.Categorical(["A", "B", None]),
                        "cluster": pd.Categorical(["0", "1", "1"])})
    result = benchmark.compute_clustering_metrics(AnnData(np.zeros((3, 2)), obs=obs), "label", "cluster")
    assert result["ARI"] == 1.
    assert result["n_evaluated_cells"] == 2


def test_no_labeled_pairs_are_an_error(benchmark):
    obs = pd.DataFrame({"label": [None, None], "cluster": ["0", "1"]})
    with pytest.raises(ValueError, match="annotat|label|pair"):
        benchmark.compute_clustering_metrics(AnnData(np.zeros((2, 2)), obs=obs), "label", "cluster")
