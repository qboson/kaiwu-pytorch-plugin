"""Optional activations in the standalone MNIST decoder examples."""

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn


def _load_networks():
    path = Path(__file__).resolve().parents[1] / "example/qvae_mnist/model/networks.py"
    spec = importlib.util.spec_from_file_location("mnist_decoder_networks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NETWORKS = _load_networks()


@pytest.mark.parametrize("decoder_class", [NETWORKS.BasicDecoder, NETWORKS.SimpleDecoder])
@pytest.mark.parametrize("activation", [None, nn.ReLU()])
@pytest.mark.parametrize("output_activation", [None, nn.Sigmoid()])
def test_optional_decoder_activations_match_explicit_layers(
    decoder_class, activation, output_activation
):
    """Missing activations are identities, including the final output layer."""
    decoder = decoder_class(
        node_sequence=[(2, 3), (3, 2)],
        activation_fct=activation,
        output_activation_fct=output_activation,
    ).double()
    with torch.no_grad():
        decoder._layers[0].weight.fill_(0.5)
        decoder._layers[0].bias.fill_(0.25)
        decoder._layers[1].weight.fill_(-0.75)
        decoder._layers[1].bias.fill_(-0.5)
    data = torch.tensor([[1., 2.], [-2., -1.]], dtype=torch.float64, requires_grad=True)
    hidden = decoder._layers[0](data)
    if activation is not None:
        hidden = activation(hidden)
    expected = decoder._layers[1](hidden)
    if output_activation is not None:
        expected = output_activation(expected)

    actual = decoder(data)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert data.grad is not None and torch.isfinite(data.grad).all()
    assert all(parameter.grad is not None for parameter in decoder.parameters())


@pytest.mark.parametrize("output_activation", [None, nn.Sigmoid()])
def test_simple_decoder_posterior_sample_uses_same_output_contract(output_activation):
    decoder = NETWORKS.SimpleDecoder(
        node_sequence=[(2, 3), (3, 2)],
        activation_fct=nn.ReLU(),
        output_activation_fct=output_activation,
    )
    data = torch.tensor([[0.5, -0.25]])
    torch.testing.assert_close(decoder.decode_posterior_sample(data), decoder(data))


@pytest.mark.parametrize("decoder_class", [NETWORKS.BasicDecoder, NETWORKS.SimpleDecoder])
def test_empty_decoder_is_identity(decoder_class):
    decoder = decoder_class()
    data = torch.tensor([[0.5, -0.25]])
    assert decoder(data) is data
