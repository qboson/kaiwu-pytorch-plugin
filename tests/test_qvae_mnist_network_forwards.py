"""Contract test for the encoder/decoder ``forward`` delegation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
NETWORKS = REPO / "example" / "qvae_mnist" / "model" / "networks.py"


def _load_networks():
    spec = importlib.util.spec_from_file_location("qvae_mnist_networks_uut", NETWORKS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _inputs():
    return torch.tensor([[1.0, -2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])


def test_basic_encoder_forward_returns_the_encoded_representation():
    networks = _load_networks()
    encoder = networks.BasicEncoder(node_sequence=[(2, 3)], activation_fct=torch.relu)
    with torch.no_grad():
        encoder._layers[0].weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        )
        encoder._layers[0].bias.zero_()
    inputs = _inputs()

    result = encoder.forward(inputs)

    assert torch.equal(result, encoder.encode(inputs))
    assert result.shape == (4, 3)
    assert not torch.equal(result, inputs)


def test_basic_decoder_forward_returns_the_decoded_representation():
    networks = _load_networks()
    decoder = networks.BasicDecoder(
        node_sequence=[(2, 3), (3, 1)],
        activation_fct=torch.relu,
        output_activation_fct=torch.sigmoid,
    )
    inputs = _inputs()

    result = decoder.forward(inputs)

    assert torch.equal(result, decoder.decode(inputs))
    assert result.shape == (4, 1)
    assert torch.all(result >= 0) and torch.all(result <= 1)
