# -*- coding: utf-8 -*-
"""plot_autoencoder_outputs must honour n and not depend on an undeclared global."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

QVAE_MNIST_ROOT = Path(__file__).resolve().parents[1] / "example" / "qvae_mnist"
if str(QVAE_MNIST_ROOT) not in sys.path:
    sys.path.insert(0, str(QVAE_MNIST_ROOT))

import utils.helpers as helpers  # noqa: E402  pylint: disable=wrong-import-position


class _IdentityDecoder:
    """Decode by returning the input tensor unchanged."""

    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        self.inputs.append(x)
        return x


def test_requested_sample_count_is_used(monkeypatch) -> None:
    """The old body overwrote n with 5 and indexed past a two-row input."""
    x_test = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    model = _IdentityDecoder()
    monkeypatch.setattr(helpers, "x_test", x_test, raising=False)
    monkeypatch.setattr(helpers.plt, "show", lambda: None)

    helpers.plot_autoencoder_outputs(model, n=2, dims=(4, 4))

    assert len(model.inputs) == 1
    assert torch.equal(model.inputs[0], x_test)
