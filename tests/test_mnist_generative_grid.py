# -*- coding: utf-8 -*-
"""The generated-image grid must not drop rows for partial last rows."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

QVAE_MNIST_ROOT = Path(__file__).resolve().parents[1] / "example" / "qvae_mnist"
if str(QVAE_MNIST_ROOT) not in sys.path:
    sys.path.insert(0, str(QVAE_MNIST_ROOT))

import utils.helpers as helpers  # noqa: E402  pylint: disable=wrong-import-position


def test_non_multiple_sample_count_renders_every_row(tmp_path) -> None:
    """Six samples need two rows; one row has only five subplot slots."""
    x_true = torch.zeros(6, 784)
    output = tmp_path / "generative_grid.png"

    helpers.plot_generative_output(x_true, n_samples=6, output=str(output))

    assert output.is_file()
    assert output.stat().st_size > 0
