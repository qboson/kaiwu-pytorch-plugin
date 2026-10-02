"""Regression test for the unsupported CellQVAE option in the MNIST example."""

import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../example/qvae_mnist")
    ),
)
from model.config import Config  # noqa: E402  pylint: disable=wrong-import-position


def test_cellqvae_is_rejected_by_config():
    """Config must not advertise a model type the pipeline cannot build."""
    with pytest.raises(ValueError, match="Unsupported model type: CellQVAE"):
        Config(model_type="CellQVAE")
