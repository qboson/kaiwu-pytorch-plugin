"""Encoding-input contract for the single-cell QVAE."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_cell"))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "qvae_cell_models_encode", os.path.join(REPO_ROOT, "example", "qvae_cell", "models.py")
)
models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(models)
CellQVAE = models.CellQVAE


class RecordingEncoder:
    """Records the tensor it receives and returns fixed logits."""

    def __init__(self):
        self.inputs = []

    def __call__(self, x):
        self.inputs.append(x)
        return torch.zeros(x.shape[0], 2)


def build_model(loss_type, dataset_mean=None):
    """Builds only what ``_encode`` reads."""
    model = CellQVAE.__new__(CellQVAE)
    model._input_dimension = 3
    model.config = SimpleNamespace(loss_type=loss_type, dist_beta=10.0)
    model._dataset_mean = dataset_mean
    model.encoder = RecordingEncoder()
    model.posterior = lambda q, beta: ("POSTERIOR", "ZETA")
    return model


def test_bernoulli_loss_centres_the_input_by_the_dataset_mean():
    """The stored dataset mean is subtracted before encoding."""
    mean = torch.tensor([1.0, 2.0, 3.0])
    model = build_model("bernoulli", mean)
    x = torch.tensor([[2.0, 2.0, 2.0]])
    q, posterior, zeta = model._encode(x)
    assert torch.equal(model.encoder.inputs[0], torch.tensor([[1.0, 0.0, -1.0]]))
    assert torch.equal(q, torch.zeros(1, 2))
    assert (posterior, zeta) == ("POSTERIOR", "ZETA")


def test_mse_loss_encodes_the_raw_input():
    """MSE reconstruction uses the uncentred features."""
    model = build_model("mse")
    x = torch.tensor([[2.0, 2.0, 2.0]])
    model._encode(x)
    assert torch.equal(model.encoder.inputs[0], x)


def test_unknown_loss_type_is_rejected():
    """Only the two documented reconstruction losses are supported."""
    model = build_model("poisson")
    with pytest.raises(ValueError, match="Unsupported loss type"):
        model._encode(torch.zeros(1, 3))
