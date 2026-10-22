"""Cell loaders share one backing dataset and keep sparse rows sparse until read."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

sparse = pytest.importorskip("scipy.sparse")
train_test_split = pytest.importorskip("sklearn.model_selection").train_test_split


def load_trainer():
    path = Path(__file__).resolve().parents[1] / "example/qvae_cell/trainer.py"
    tree = ast.parse(path.read_text())
    definitions = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Trainer":
            node.body = [item for item in node.body if isinstance(item, ast.FunctionDef)
                         and item.name in {"__init__", "adata_to_array", "make_loaders", "compute_mean_x"}]
            definitions.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "_SparseExpressionDataset":
            definitions.append(node)
    module = ast.Module(body=definitions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(np=np, torch=torch, DataLoader=DataLoader, TensorDataset=TensorDataset,
                     Dataset=Dataset, Subset=Subset, train_test_split=train_test_split)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["Trainer"](SimpleNamespace(
        loss_type="mse", val_percentage=.25, seed=42, batch_size=4), "cpu")


@pytest.mark.parametrize("use_sparse", [False, True])
def test_loaders_share_backing_data_and_preserve_split_and_mean(use_sparse):
    trainer = load_trainer()
    values = np.arange(48, dtype=np.float32).reshape(12, 4)
    matrix = sparse.csr_matrix(values) if use_sparse else values
    prepared = trainer.adata_to_array(SimpleNamespace(X=matrix))
    if use_sparse:
        assert sparse.issparse(prepared)
    train, val, evaluation = trainer.make_loaders(prepared, np.arange(12) % 3)
    assert train.dataset.dataset is evaluation.dataset
    assert val.dataset.dataset is evaluation.dataset
    train_ids, val_ids = train_test_split(np.arange(12), test_size=.25, random_state=42)
    assert list(train.dataset.indices) == list(train_ids)
    assert list(val.dataset.indices) == list(val_ids)
    for index in range(12):
        row, batch = evaluation.dataset[index]
        torch.testing.assert_close(row, torch.from_numpy(values[index]))
        assert batch.item() == index % 3
    np.testing.assert_allclose(trainer.compute_mean_x(train), values[train_ids].mean(0))
    if not use_sparse:
        assert evaluation.dataset.tensors[0].data_ptr() == values.ctypes.data


def test_sparse_mse_preparation_and_loading_never_densifies_the_whole_matrix(monkeypatch):
    trainer = load_trainer()
    matrix = sparse.csr_matrix(np.eye(12, 4, dtype=np.float32))
    original = sparse.csr_matrix.toarray

    def row_only(self, *args, **kwargs):
        assert self.shape[0] == 1, "full expression matrix was densified"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(sparse.csr_matrix, "toarray", row_only)
    prepared = trainer.adata_to_array(SimpleNamespace(X=matrix))
    train, _, _ = trainer.make_loaders(prepared, np.zeros(12, dtype=int))
    batch, _ = next(iter(train))
    assert batch.shape == (4, 4)
