"""Unannotated cells cannot become synthetic supervised classes."""
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    any(find_spec(name) is None for name in ("anndata", "scanpy")),
    reason="requires the single-cell example dependencies",
)


def make_data(labels):
    import anndata
    adata = anndata.AnnData(np.zeros((len(labels), 2), dtype=np.float32))
    adata.obs["label"] = pd.Categorical(labels)
    adata.obsm["X_qvae"] = np.array(
        [[i % 2, (i % 2) * 10] for i in range(len(labels))], dtype=np.float32
    )
    return adata


def test_missing_annotations_are_filtered_before_splitting():
    from example.qvae_cell.evaluate_benchmark import evaluate_classifier
    labels = ["A", "nan"] * 20 + [None, np.nan, None, np.nan]
    adata = make_data(labels)
    original = adata.copy()
    labeled = adata[:40].copy()
    actual = evaluate_classifier(adata, "X_qvae", "label", .25, 42, 100)
    expected = evaluate_classifier(labeled, "X_qvae", "label", .25, 42, 100)
    assert actual.loc[0, "n_classes"] == 2  # literal 'nan' remains a real label
    assert actual.loc[0, "n_labeled_cells"] == 40
    assert actual.loc[0, "n_missing_labels"] == 4
    for metric in ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted"):
        assert actual.loc[0, metric] == expected.loc[0, metric]
    pd.testing.assert_frame_equal(adata.obs, original.obs)
    np.testing.assert_array_equal(adata.obsm["X_qvae"], original.obsm["X_qvae"])


def test_no_annotations_has_actionable_error():
    from example.qvae_cell.evaluate_benchmark import evaluate_classifier
    with pytest.raises(ValueError, match="non-missing"):
        evaluate_classifier(make_data([None] * 12), "X_qvae", "label", .25, 42, 100)
