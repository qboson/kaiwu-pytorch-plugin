"""The cell-type benchmark supports sparse as well as dense embeddings."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

anndata = pytest.importorskip("anndata")
pytest.importorskip("scanpy")

from example.qvae_cell.evaluate_benchmark import evaluate_classifier


def test_sparse_embedding_produces_the_same_heldout_metrics_as_dense():
    labels = np.repeat(["early", "late"], 12)
    embedding = np.repeat(np.eye(2), 12, axis=0)
    cells = anndata.AnnData(
        np.zeros((len(labels), 1)), obs=pd.DataFrame({"label": labels}, index=map(str, range(24))),
    )
    cells.obsm["dense"] = embedding
    cells.obsm["sparse"] = sparse.csr_matrix(embedding)
    kwargs = {"label_key": "label", "test_size": 0.25, "random_state": 42, "max_iter": 100}

    dense_result = evaluate_classifier(cells, "dense", **kwargs)
    sparse_result = evaluate_classifier(cells, "sparse", **kwargs)

    assert dense_result.loc[0, "accuracy"] == 1.0
    pd.testing.assert_frame_equal(dense_result.drop(columns="rep_key"), sparse_result.drop(columns="rep_key"))
