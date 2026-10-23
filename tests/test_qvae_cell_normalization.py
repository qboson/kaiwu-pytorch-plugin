"""Normalization-validation contract for the single-cell QVAE networks."""

import os
import sys

import importlib.util

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "qvae_cell_models_norm", os.path.join(REPO_ROOT, "example", "qvae_cell", "models.py")
)
models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(models)
QVAEDecoder = models.QVAEDecoder
QVAEEncoder = models.QVAEEncoder


@pytest.mark.parametrize("normalization", ["bogus", "Layered", ""])
def test_encoder_rejects_unknown_normalization(normalization):
    """Only ``layer`` and ``batch`` are accepted by the encoder."""
    with pytest.raises(ValueError, match="normalization_method"):
        QVAEEncoder(4, 6, 3, normalization_method=normalization)


@pytest.mark.parametrize("normalization", ["bogus", "Layered", ""])
def test_decoder_rejects_unknown_normalization(normalization):
    """The decoder applies the same whitelist."""
    with pytest.raises(ValueError, match="normalization_method"):
        QVAEDecoder(3, 6, 4, normalization_method=normalization)


def test_layer_and_batch_norms_are_selected_by_name():
    """Each accepted name builds the matching normalization module."""
    layer_encoder = QVAEEncoder(4, 6, 3, normalization_method="layer")
    batch_encoder = QVAEEncoder(4, 6, 3, normalization_method="batch")
    assert isinstance(layer_encoder.norm, torch.nn.LayerNorm)
    assert isinstance(batch_encoder.norm, torch.nn.BatchNorm1d)


def test_forward_shapes_follow_the_declared_dimensions():
    """Encoder maps features to latent logits; decoder maps back to features."""
    encoder = QVAEEncoder(4, 6, 3, normalization_method="layer")
    decoder = QVAEDecoder(3, 6, 4, normalization_method="layer")
    encoder.eval()
    decoder.eval()
    latents = encoder(torch.zeros(2, 4))
    recon = decoder(latents)
    assert latents.shape == (2, 3)
    assert recon.shape == (2, 4)
