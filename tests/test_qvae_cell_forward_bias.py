"""Forward-path contract for the single-cell QVAE reconstruction."""

import importlib.util
import os

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "qvae_cell_models_forward", os.path.join(REPO_ROOT, "example", "qvae_cell", "models.py")
)
models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(models)
CellQVAE = models.CellQVAE

from types import SimpleNamespace  # noqa: E402


def build_model(loss_type):
    """Builds only what ``forward`` reads, with a zero reconstruction."""
    model = CellQVAE.__new__(CellQVAE)
    model.config = SimpleNamespace(loss_type=loss_type)
    model._train_bias = torch.full((2,), 0.5)
    model._encode = lambda x: ("Q", "POSTERIOR", "ZETA")
    model._decoder_input = lambda zeta, batch_idx=None: zeta
    model.decoder = lambda zeta: torch.zeros(1, 2)
    return model


def test_bernoulli_reconstruction_adds_the_training_bias():
    """The stored train bias shifts the decoder output for Bernoulli loss."""
    recon, posterior, q, zeta = build_model("bernoulli").forward(torch.zeros(1, 2))
    assert recon.tolist() == [[0.5, 0.5]]
    assert (posterior, q, zeta) == ("POSTERIOR", "Q", "ZETA")


def test_mse_reconstruction_keeps_the_decoder_output():
    """Continuous reconstruction is returned without the Bernoulli bias."""
    recon, _posterior, _q, _zeta = build_model("mse").forward(torch.zeros(1, 2))
    assert recon.tolist() == [[0.0, 0.0]]
