"""BasicDecoder must tolerate activation_fct=None like the encoder does."""

import importlib.util
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORKS_PATH = REPO_ROOT / "example" / "qvae_mnist" / "model" / "networks.py"


def _load():
    spec = importlib.util.spec_from_file_location("qvae_mnist_networks", NETWORKS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decoder_without_activation_returns_linear_output() -> None:
    networks = _load()
    decoder = networks.BasicDecoder(
        node_sequence=[(2, 3), (3, 2)],
        activation_fct=None,
        output_activation_fct=torch.nn.Identity(),
    )
    inputs = torch.randn(4, 2)

    output = decoder(inputs)

    expected = decoder._layers[1](decoder._layers[0](inputs))  # pylint: disable=protected-access
    torch.testing.assert_close(output, expected)
    assert output.shape == (4, 2)


def test_decoder_with_activation_still_uses_it() -> None:
    networks = _load()
    decoder = networks.BasicDecoder(
        node_sequence=[(2, 3), (3, 2)],
        activation_fct=torch.nn.ReLU(),
        output_activation_fct=torch.nn.Identity(),
    )
    inputs = torch.randn(4, 2)

    output = decoder(inputs)

    hidden = torch.relu(decoder._layers[0](inputs))  # pylint: disable=protected-access
    expected = decoder._layers[1](hidden)  # pylint: disable=protected-access
    torch.testing.assert_close(output, expected)
