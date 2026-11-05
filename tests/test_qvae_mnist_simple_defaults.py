"""Contract test for the simplified encoder/decoder constructor defaults."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
NETWORKS = REPO / "example" / "qvae_mnist" / "model" / "networks.py"


def _load_networks():
    spec = importlib.util.spec_from_file_location("qvae_mnist_networks_defaults_uut", NETWORKS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_simple_encoder_keeps_the_documented_architecture():
    networks = _load_networks()

    encoder = networks.SimpleEncoder(smoothing_distribution="exponential")

    assert encoder.smoothing_distribution == "exponential"
    assert encoder.num_latent_hierarchy_levels == 4
    assert encoder.num_latent_units == 100
    assert encoder.num_det_units == 200
    assert encoder.num_det_layers == 2


def test_simple_decoder_keeps_the_output_activation():
    networks = _load_networks()

    decoder = networks.SimpleDecoder(output_activation_fct=torch.sigmoid)

    assert decoder._output_activation_fct is torch.sigmoid
    assert networks.SimpleDecoder()._output_activation_fct is None
