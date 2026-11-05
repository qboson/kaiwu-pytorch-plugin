# -*- coding: utf-8 -*-
"""Unsupported MNIST variants should fail with the valid choices, not KeyError."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

QVAE_MNIST_ROOT = Path(__file__).resolve().parents[1] / "example" / "qvae_mnist"
if str(QVAE_MNIST_ROOT) not in sys.path:
    sys.path.insert(0, str(QVAE_MNIST_ROOT))

from utils.loadMNIST import loadMNIST  # noqa: E402  pylint: disable=wrong-import-position


def test_unknown_dataset_name_is_rejected_before_download() -> None:
    with pytest.raises(ValueError, match="Unsupported dataset"):
        loadMNIST(name="not-a-dataset")
