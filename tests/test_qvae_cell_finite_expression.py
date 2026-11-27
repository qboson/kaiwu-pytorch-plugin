"""Reject invalid expression values before creating a model or sampler."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def trainer_class(monkeypatch):
    pytest.importorskip("sklearn", exc_type=ImportError)
    directory = Path(__file__).resolve().parents[1] / "example" / "qvae_cell"
    for name in ("models", "trainer"):
        spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module.Trainer


@pytest.mark.parametrize("loss_type", ["mse", "bernoulli"])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, 1e100])
def test_invalid_expression_fails_in_data_preparation(trainer_class, loss_type, sparse, invalid):
    values = np.array([[1.0, invalid], [0.0, 2.0]], dtype=np.float64)
    if sparse:
        values = pytest.importorskip("scipy.sparse", exc_type=ImportError).csr_matrix(values)
    trainer = trainer_class(SimpleNamespace(loss_type=loss_type), torch.device("cpu"))

    with pytest.raises(ValueError, match=r"AnnData\.X.*finite float32"):
        trainer.adata_to_array(SimpleNamespace(X=values))


@pytest.mark.parametrize("loss_type", ["mse", "bernoulli"])
def test_finite_sparse_expression_keeps_existing_preprocessing(trainer_class, loss_type):
    values = np.array([[-2.0, 0.0], [2.0, 6.0]])
    sparse = pytest.importorskip("scipy.sparse", exc_type=ImportError).csr_matrix(values)
    trainer = trainer_class(SimpleNamespace(loss_type=loss_type), torch.device("cpu"))
    expected = values if loss_type == "mse" else (values + 2.0) / 8.0

    actual = trainer.adata_to_array(SimpleNamespace(X=sparse))
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_large_finite_range_stays_finite_after_bernoulli_scaling(trainer_class):
    limit = np.finfo(np.float32).max
    values = np.array([[-limit, 0, limit]], dtype=np.float32)
    trainer = trainer_class(SimpleNamespace(loss_type="bernoulli"), torch.device("cpu"))
    np.testing.assert_array_equal(
        trainer.adata_to_array(SimpleNamespace(X=values)), [[0, 0.5, 1]]
    )
