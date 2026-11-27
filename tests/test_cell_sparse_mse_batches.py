"""Sparse cell MSE batches retain the dense input/split/label contracts."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sparse = pytest.importorskip("scipy.sparse")
anndata = pytest.importorskip("anndata")
pytest.importorskip("sklearn")
pytest.importorskip("kaiwu")


@pytest.fixture
def trainer_class():
    directory = Path(__file__).resolve().parents[1] / "example/qvae_cell"
    previous_models = sys.modules.pop("models", None)
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("cell_sparse_batches", directory / "trainer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        sys.modules.pop("models", None)
        if previous_models is not None:
            sys.modules["models"] = previous_models
    return module.Trainer


def make_trainer(trainer_class, loss_type="mse"):
    args = SimpleNamespace(loss_type=loss_type, val_percentage=.2, seed=42, batch_size=4)
    return trainer_class(args, torch.device("cpu"))


def check_batches_and_mean(trainer, prepared, expected):
    codes = np.arange(len(expected)) % 3
    loaders = trainer.make_loaders(prepared, codes)
    permutation = np.random.RandomState(42).permutation(len(expected))
    n_val = int(np.ceil(.2 * len(expected)))
    splits = (permutation[n_val:], permutation[:n_val], np.arange(len(expected)))
    with torch.random.fork_rng(devices=[]):
        for index, (loader, ids) in enumerate(zip(loaders, splits)):
            reference = DataLoader(TensorDataset(torch.from_numpy(ids)), batch_size=4, shuffle=index == 0)
            torch.manual_seed(81)
            row_batches = [rows[0].numpy() for rows in reference]
            torch.manual_seed(81)
            observed = list(loader)
            assert len(observed) == len(row_batches)
            for (values, labels), rows in zip(observed, row_batches):
                assert values.dtype == torch.float32
                np.testing.assert_array_equal(values.numpy(), expected[rows])
                np.testing.assert_array_equal(labels.numpy(), codes[rows])
        actual_mean = trainer.compute_mean_x(loaders[0])
    np.testing.assert_allclose(actual_mean, expected[splits[0]].mean(axis=0), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("format_name", ["csr", "csc"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_sparse_mse_preserves_every_batch_and_the_mean(trainer_class, format_name, dtype):
    expected = np.arange(105).reshape(15, 7).astype(dtype)
    expected[expected % 3 != 0] = 0
    matrix = getattr(sparse, format_name + "_matrix")(expected)
    trainer = make_trainer(trainer_class)

    prepared = trainer.adata_to_array(anndata.AnnData(matrix))

    assert sparse.isspmatrix_csr(prepared)
    assert prepared.dtype == matrix.dtype
    check_batches_and_mean(trainer, prepared, expected.astype(np.float32))


def test_sparse_materialization_is_bounded_by_a_batch(trainer_class, monkeypatch):
    matrix = sparse.csr_matrix(np.eye(15, 7, dtype=np.float32))
    original = sparse.csr_matrix.toarray
    calls = []

    def batch_only(self, *args, **kwargs):
        assert self.shape[0] <= 4, "the entire expression matrix was materialized"
        calls.append(self.shape[0])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(sparse.csr_matrix, "toarray", batch_only)
    trainer = make_trainer(trainer_class)
    prepared = trainer.adata_to_array(anndata.AnnData(matrix))
    loaders = trainer.make_loaders(prepared, np.zeros(15, dtype=np.int64))
    for loader in loaders:
        for values, _ in loader:
            assert values.shape[0] <= 4
    assert calls and max(calls) == 4


@pytest.mark.parametrize("loss_type", ["mse", "bernoulli"])
@pytest.mark.parametrize("sparse_input", [False, True])
def test_dense_and_bernoulli_preprocessing_values_are_preserved(trainer_class, loss_type, sparse_input):
    original = np.tile(np.array([-2, 1, 4], dtype=np.float32), (15, 1))
    matrix = sparse.csr_matrix(original) if sparse_input else original.copy()
    trainer = make_trainer(trainer_class, loss_type)

    prepared = trainer.adata_to_array(anndata.AnnData(matrix))

    expected = np.tile(np.array([0, .5, 1], dtype=np.float32), (15, 1)) if loss_type == "bernoulli" else original
    if not sparse_input or loss_type == "bernoulli":
        assert isinstance(prepared, np.ndarray)
        np.testing.assert_array_equal(prepared, expected)
    check_batches_and_mean(trainer, prepared, expected)
    np.testing.assert_array_equal(original, np.tile(np.array([-2, 1, 4], dtype=np.float32), (15, 1)))


@pytest.mark.parametrize("format_name", ["csr", "csc"])
def test_duplicate_float64_entries_accumulate_before_float32_conversion(trainer_class, format_name):
    data = np.tile(np.array([1.00000006, -1.], dtype=np.float64), 15)
    if format_name == "csr":
        matrix = sparse.csr_matrix((data, np.zeros(30, dtype=np.int32), np.arange(0, 31, 2)), shape=(15, 3))
    else:
        matrix = sparse.csc_matrix((data, np.repeat(np.arange(15), 2), [0, 30, 30, 30]), shape=(15, 3))
    # This is the original eager pipeline's arithmetic and conversion order.
    expected = matrix.toarray().astype(np.float32)
    original_data = matrix.data.copy()
    trainer = make_trainer(trainer_class)

    prepared = trainer.adata_to_array(anndata.AnnData(matrix))

    check_batches_and_mean(trainer, prepared, expected)
    np.testing.assert_array_equal(matrix.data, original_data)
