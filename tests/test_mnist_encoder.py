"""Regression tests for the MNIST QVAE encoder output activation."""

import os
import sys

import torch
from torch import nn

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../example/qvae_mnist"))
)
from model.config import Config  # noqa: E402  pylint: disable=wrong-import-position
from model.model import MnistQVAE  # noqa: E402  pylint: disable=wrong-import-position
from model.networks import BasicEncoder  # noqa: E402  pylint: disable=wrong-import-position


def test_basic_encoder_applies_output_activation_to_last_layer_only():
    encoder = BasicEncoder(
        node_sequence=[(3, 4), (4, 2)],
        activation_fct=nn.ReLU(),
        output_activation_fct=nn.Identity(),
    )
    with torch.no_grad():
        for layer in encoder._layers:  # pylint: disable=protected-access
            layer.weight.fill_(1.0)
            layer.bias.zero_()
        encoder._layers[-1].weight.fill_(-1.0)  # pylint: disable=protected-access

    output = encoder(torch.ones(1, 3))

    assert torch.all(output < 0)


def test_mnist_qvae_encoder_returns_logits_not_rectified_values():
    """The latent logits must be able to encode p(z=1) < 0.5."""
    config = Config("QVAE")
    model = MnistQVAE(input_dimension=8, activation_fct=config.activation_fct, config=config)
    encoder = model.encoder
    with torch.no_grad():
        encoder._layers[0].weight.fill_(1.0)  # pylint: disable=protected-access
        encoder._layers[0].bias.zero_()  # pylint: disable=protected-access
        encoder._layers[-1].weight.fill_(-1.0)  # pylint: disable=protected-access
        encoder._layers[-1].bias.zero_()  # pylint: disable=protected-access

    logits = encoder(torch.ones(2, 8))

    assert torch.all(logits < 0), "encoder output is rectified, posterior is stuck at >= 0.5"
