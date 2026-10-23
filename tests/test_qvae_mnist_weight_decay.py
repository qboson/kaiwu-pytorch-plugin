"""Weight-decay contract for the QVAE-MNIST encoder."""

import os
import sys

import importlib.util

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "qvae_mnist_networks",
    os.path.join(REPO_ROOT, "example", "qvae_mnist", "model", "networks.py"),
)
networks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(networks)
BasicEncoder = networks.BasicEncoder


def test_zero_weight_decay_returns_a_tensor_on_the_parameter_device():
    """The zero sentinel is a tensor so callers can add it to a loss."""
    encoder = BasicEncoder(
        node_sequence=[[2, 3]],
        activation_fct=torch.nn.ReLU(),
        weight_decay=0.0,
    )
    value = encoder.get_weight_decay()
    assert isinstance(value, torch.Tensor)
    assert value.item() == 0.0
    assert value.device == next(encoder.parameters()).device


def test_weight_decay_scales_the_sum_of_squared_weights():
    """Only linear weights contribute, scaled by the configured coefficient."""
    encoder = BasicEncoder(
        node_sequence=[[2, 3], [3, 1]],
        activation_fct=None,
        weight_decay=0.5,
    )
    with torch.no_grad():
        for layer in encoder._layers:
            layer.weight.fill_(2.0)
            layer.bias.zero_()
    # 6 + 3 = 9 weight elements, each squared to 4 -> 36
    assert encoder.get_weight_decay().item() == pytest.approx(0.5 * 36.0)
