"""Encoder-construction contract for the MNIST QVAE."""

import os
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))

from model.model import MnistQVAE  # noqa: E402
from model.networks import BasicEncoder  # noqa: E402


def build_model():
    """Builds only the attributes ``_create_encoder`` reads."""
    model = MnistQVAE.__new__(MnistQVAE)
    model._input_dimension = 8
    model._latent_dimensions = 4
    model._activation_fct = torch.relu
    model.weight_decay = 0.25
    model.config = SimpleNamespace(encoder_hidden_nodes=[6])
    return model


def test_encoder_layers_walk_from_input_to_latent():
    """The node pairs are input -> hidden -> latent."""
    encoder = build_model()._create_encoder()
    assert isinstance(encoder, BasicEncoder)
    assert [
        (layer.in_features, layer.out_features) for layer in encoder._layers
    ] == [(8, 6), (6, 4)]


def test_encoder_carries_the_weight_decay():
    """The model's weight decay reaches the encoder."""
    encoder = build_model()._create_encoder()
    assert encoder.weight_decay == 0.25
