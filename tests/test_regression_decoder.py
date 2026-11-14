"""Exercise the real sequential decoder, its gradients and saved state."""
import importlib.util
from pathlib import Path
import pytest
import torch
from torch import nn


def networks():
    path = Path(__file__).resolve().parents[1] / "example/qvae_mnist/model/networks.py"
    spec = importlib.util.spec_from_file_location("mnist_decoder_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("output_activation", [None, nn.Sigmoid()])
def test_sequential_decoder_builds_and_matches_independent_network(output_activation):
    module = networks()
    torch.manual_seed(71)
    decoder = module.Decoder(node_sequence=[(3, 4), (4, 2)],
                             activation_fct=nn.Tanh(),
                             output_activation_fct=output_activation, weight_decay=0.2)
    linears = [layer for layer in decoder.modules() if isinstance(layer, nn.Linear)]
    assert len(linears) == 2
    inputs = torch.randn(5, 3, requires_grad=True)
    expected = linears[1](torch.tanh(linears[0](inputs)))
    if output_activation is not None:
        expected = output_activation(expected)
    torch.testing.assert_close(decoder(inputs), expected)
    expected_penalty = 0.2 * sum(layer.weight.square().sum() for layer in linears)
    torch.testing.assert_close(decoder.get_weight_decay(), expected_penalty)
    (decoder(inputs).square().sum() + decoder.get_weight_decay()).backward()
    assert all(parameter.grad is not None for parameter in decoder.parameters())
    clone = module.Decoder(node_sequence=[(3, 4), (4, 2)], activation_fct=nn.Tanh(),
                           output_activation_fct=output_activation, weight_decay=0.2)
    clone.load_state_dict(decoder.state_dict())
    torch.testing.assert_close(clone(inputs), decoder(inputs))


def test_default_decoder_is_identity_and_zero_regularization():
    decoder = networks().Decoder()
    inputs = torch.randn(2, 3)
    torch.testing.assert_close(decoder(inputs), inputs)
    assert decoder.get_weight_decay() == 0


@pytest.mark.parametrize("create_module_list", [True, False])
def test_sequential_decoder_accepts_inherited_construction_option(create_module_list):
    decoder = networks().Decoder(node_sequence=[(3, 2)],
                                  create_module_list=create_module_list)
    assert decoder(torch.ones(1, 3)).shape == (1, 2)
    assert len(list(decoder.parameters())) == 2


def test_basic_decoder_behavior_is_unchanged():
    decoder = networks().BasicDecoder(node_sequence=[(2, 2)],
                                     activation_fct=nn.ReLU(), output_activation_fct=nn.Identity())
    with torch.no_grad():
        decoder._layers[0].weight.copy_(-torch.eye(2))
        decoder._layers[0].bias.zero_()
    torch.testing.assert_close(decoder(torch.ones(1, 2)), -torch.ones(1, 2))
