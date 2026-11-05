"""Contract test for the qvae_cell encoder/decoder forward passes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "example" / "qvae_cell" / "models.py"


def _load_models():
    spec = importlib.util.spec_from_file_location("qvae_cell_models_uut", MODELS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_encoder_returns_the_latent_logits():
    models = _load_models()
    encoder = models.QVAEEncoder(input_dim=6, hidden_dim=4, latent_dim=3)

    logits = encoder(torch.zeros(5, 6))

    assert logits.shape == (5, 3)
    assert torch.isfinite(logits).all()


def test_decoder_returns_the_reconstruction_width():
    models = _load_models()
    decoder = models.QVAEDecoder(latent_dim=3, hidden_dim=4, output_dim=7)

    reconstruction = decoder(torch.zeros(2, 3))

    assert reconstruction.shape == (2, 7)
    assert torch.isfinite(reconstruction).all()


def test_batch_normalisation_encoders_accept_a_real_batch():
    models = _load_models()
    encoder = models.QVAEEncoder(
        input_dim=4, hidden_dim=3, latent_dim=2, normalization_method="batch"
    )
    encoder.train()

    logits = encoder(torch.randn(6, 4))

    assert logits.shape == (6, 2)
