"""Layer-construction contract for the QVAE-MNIST networks."""

import importlib.util
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "qvae_mnist_networks_layers",
    os.path.join(REPO_ROOT, "example", "qvae_mnist", "model", "networks.py"),
)
networks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(networks)
BasicEncoder = networks.BasicEncoder


def test_node_sequence_builds_one_linear_layer_per_pair():
    """Each ``[in, out]`` pair becomes a Linear with those dimensions."""
    encoder = BasicEncoder(
        node_sequence=[[4, 3], [3, 2]],
        activation_fct=torch.nn.ReLU(),
    )
    assert len(encoder._layers) == 2
    assert (encoder._layers[0].in_features, encoder._layers[0].out_features) == (4, 3)
    assert (encoder._layers[1].in_features, encoder._layers[1].out_features) == (3, 2)


def test_without_a_node_sequence_no_layers_are_created():
    """An empty sequence leaves the module list empty."""
    encoder = BasicEncoder(node_sequence=None, activation_fct=torch.nn.ReLU())
    assert len(encoder._layers) == 0


def test_activation_name_drops_the_constructor_parentheses():
    """The reported activation name is the class name without ``()``."""
    encoder = BasicEncoder(
        node_sequence=[[2, 2]],
        activation_fct=torch.nn.ReLU(),
    )
    assert encoder.get_activation_fct() == "ReLU"


def test_forward_runs_through_every_layer():
    """A two-layer encoder maps the declared input width to the output width."""
    encoder = BasicEncoder(
        node_sequence=[[4, 3], [3, 2]],
        activation_fct=None,
    )
    output = encoder(torch.zeros(5, 4))
    assert output.shape == (5, 2)
