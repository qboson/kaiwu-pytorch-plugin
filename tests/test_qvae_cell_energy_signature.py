"""CellQVAE.energy must follow the base QVAE signature (loss_type optional)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "example" / "qvae_cell")
)

from models import CellQVAE  # noqa: E402  pylint: disable=wrong-import-position
from kaiwu.torch_plugin import BoltzmannMachine  # noqa: E402  pylint: disable=wrong-import-position


class _Encoder(torch.nn.Module):
    def forward(self, inputs):
        return torch.zeros(inputs.size(0), 4)


class _Decoder(torch.nn.Module):
    def forward(self, zeta):
        return torch.zeros(zeta.size(0), 6)


class _Sampler:  # pylint: disable=too-few-public-methods
    def solve(self, ising_matrix):
        return np.ones((1, ising_matrix.shape[0]), dtype=np.float32)


def _model() -> CellQVAE:
    config = SimpleNamespace(
        num_latent_units=4,
        loss_type="mse",
        dist_beta=1.0,
        kl_beta=1.0,
        weight_decay=0.0,
    )
    return CellQVAE(
        6,
        None,
        config,
        encoder=_Encoder(),
        decoder=_Decoder(),
        bm=BoltzmannMachine(4),
        sampler=_Sampler(),
        n_batches=0,
    )


def test_energy_uses_the_configured_loss_type_by_default() -> None:
    energy = _model().energy(torch.rand(3, 6))
    assert energy.shape == (3,)
    assert torch.isfinite(energy).all()


def test_energy_still_rejects_a_mismatched_loss_type() -> None:
    with pytest.raises(ValueError, match="loss_type must match"):
        _model().energy(torch.rand(2, 6), "bernoulli")
