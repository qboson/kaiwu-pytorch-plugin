"""Regression tests for the latent logits produced by the MNIST QVAE encoder."""

import importlib
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "example" / "qvae_mnist"


@pytest.fixture
def mnist_model(monkeypatch):
    """Import the example's ``model`` package from its own directory."""
    monkeypatch.syspath_prepend(str(EXAMPLE_ROOT))
    for name in [m for m in sys.modules if m == "model" or m.startswith("model.")]:
        monkeypatch.delitem(sys.modules, name)
    return importlib.import_module("model")


def test_mnist_qvae_encoder_returns_unrectified_logits(mnist_model):
    """The last encoder layer feeds sigmoid(q), so it must not pass through ReLU."""
    torch.manual_seed(0)
    config = mnist_model.Config("QVAE")
    model = mnist_model.MnistQVAE(
        input_dimension=16, activation_fct=config.activation_fct, config=config
    )
    hidden, last = model.encoder._layers  # pylint: disable=protected-access
    x = torch.rand(32, 16)

    with torch.no_grad():
        q = model.encoder(x)
        expected = last(torch.relu(hidden(x)))

    torch.testing.assert_close(q, expected)
    assert (torch.sigmoid(q) < 0.5).any()


def test_basic_encoder_without_output_activation_keeps_previous_behavior(mnist_model):
    """Callers that do not pass output_activation_fct still activate every layer."""
    torch.manual_seed(0)
    encoder = mnist_model.BasicEncoder(
        node_sequence=[(6, 5), (5, 4)], activation_fct=nn.ReLU()
    )
    x = torch.randn(8, 6)
    first, second = encoder._layers  # pylint: disable=protected-access

    with torch.no_grad():
        torch.testing.assert_close(
            encoder(x), torch.relu(second(torch.relu(first(x))))
        )
