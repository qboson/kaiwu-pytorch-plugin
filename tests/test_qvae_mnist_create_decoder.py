"""Decoder-construction contract for the MNIST QVAE."""

import os
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))

from model.model import MnistQVAE  # noqa: E402
from model.networks import BasicDecoder  # noqa: E402


def build_model():
    """Builds only the attributes ``_create_decoder`` reads."""
    model = MnistQVAE.__new__(MnistQVAE)
    model._input_dimension = 8
    model._latent_dimensions = 4
    model._activation_fct = torch.relu
    model.weight_decay = 0.25
    model.config = SimpleNamespace(decoder_hidden_nodes=[6])
    return model


def test_decoder_layers_walk_from_latent_to_input_dimension():
    """The node pairs are latent -> hidden -> input."""
    decoder = build_model()._create_decoder()
    assert isinstance(decoder, BasicDecoder)
    assert [
        (layer.in_features, layer.out_features) for layer in decoder._layers
    ] == [(4, 6), (6, 8)]


def test_decoder_outputs_logits_and_carries_the_weight_decay():
    """The output activation is the identity and the decay coefficient is kept."""
    decoder = build_model()._create_decoder()
    assert isinstance(decoder._output_activation_fct, torch.nn.Identity)
    assert decoder.weight_decay == 0.25
