"""Contract test for ``BasicDecoder.get_weight_decay``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
NETWORKS = REPO / "example" / "qvae_mnist" / "model" / "networks.py"


def _load_networks():
    spec = importlib.util.spec_from_file_location(
        "qvae_mnist_networks_decoder_wd_uut", NETWORKS
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _decoder(module, weight_decay):
    return module.BasicDecoder(
        node_sequence=[(2, 3), (3, 1)],
        activation_fct=torch.relu,
        weight_decay=weight_decay,
    )


def test_scales_the_sum_of_squared_linear_weights():
    networks = _load_networks()
    decoder = _decoder(networks, 0.5)
    expected = 0.5 * sum(
        torch.sum(layer.weight**2)
        for layer in decoder._layers
        if isinstance(layer, torch.nn.Linear)
    )

    value = decoder.get_weight_decay()

    assert torch.is_tensor(value)
    assert value.ndim == 0
    assert torch.allclose(value, expected)


def test_zero_weight_decay_returns_a_zero_tensor_on_the_parameter_device():
    networks = _load_networks()
    decoder = _decoder(networks, 0.0)

    value = decoder.get_weight_decay()

    assert float(value) == 0.0
    assert value.device == next(decoder.parameters()).device
