# -*- coding: utf-8 -*-
"""QVAE sample generation must use the same smoothing mixture as DVAE#."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

QVAE_MNIST_ROOT = Path(__file__).resolve().parents[1] / "example" / "qvae_mnist"
if str(QVAE_MNIST_ROOT) not in sys.path:
    sys.path.insert(0, str(QVAE_MNIST_ROOT))

import utils.helpers as helpers  # noqa: E402  pylint: disable=wrong-import-position


class _Sampler:
    """Placeholder sampler; the BM below ignores it."""


class _FixedBM:
    """Return a fixed 0/1 latent code so the mixture can be asserted."""

    def sample(self, sampler):
        del sampler
        return torch.tensor([[0.0, 1.0, 0.0, 1.0]])


class _RecordingDecoder(torch.nn.Module):
    """Record the latent mixture fed to the decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, zeta: torch.Tensor) -> torch.Tensor:
        self.inputs.append(zeta.detach().clone())
        return torch.zeros(zeta.shape[0], 784)


class _Model:
    """Minimal model surface used by generate_qvae_samples."""

    dist_beta = 1.0
    _train_bias = torch.zeros(784)

    def __init__(self) -> None:
        self.bm = _FixedBM()
        self.decoder = _RecordingDecoder()

    def eval(self):
        return self


class _ConstantExponential:
    """Deterministic stand-in for torch.distributions.Exponential."""

    def __init__(self, beta: float) -> None:
        self.beta = beta

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.full(shape, 0.25)


def test_sample_generation_maps_one_latent_units_to_one_minus_zeta(monkeypatch) -> None:
    """z == 1 selects the mirrored exponential sample, not a hard zero."""
    monkeypatch.setattr(helpers, "init_qvae_sampler", lambda model: _Sampler())
    monkeypatch.setattr(helpers, "Exponential", _ConstantExponential)
    model = _Model()

    helpers.generate_qvae_samples(
        model,
        dist_beta=1.0,
        n_images=1,
        batch_size=1,
    )

    assert len(model.decoder.inputs) == 1
    assert torch.allclose(
        model.decoder.inputs[0],
        torch.tensor([[0.25, 0.75, 0.25, 0.75]]),
    )
