"""QVAE.bm_loss must validate its weight-decay coefficient."""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine, QVAE


class _Encoder(torch.nn.Module):
    def forward(self, inputs):
        return torch.zeros(inputs.size(0), 4)


class _Decoder(torch.nn.Module):
    def forward(self, zeta):
        return torch.zeros(zeta.size(0), 6)


class _Sampler:  # pylint: disable=too-few-public-methods
    def solve(self, ising_matrix):
        return np.ones((1, ising_matrix.shape[0]), dtype=np.float32)


class _QVAE(QVAE):  # pylint: disable=too-few-public-methods
    def _create_encoder(self):
        return self.encoder

    def _create_decoder(self):
        return self.decoder

    def _create_bm(self):
        return self.bm

    def _create_sampler(self, sampler_type):
        del sampler_type
        return self.sampler


def _model() -> _QVAE:
    return _QVAE(
        6,
        None,
        types.SimpleNamespace(
            num_latent_units=4,
            loss_type="bernoulli",
            dist_beta=1.0,
            kl_beta=1.0,
            weight_decay=0.0,
        ),
        encoder=_Encoder(),
        decoder=_Decoder(),
        bm=BoltzmannMachine(4),
        sampler=_Sampler(),
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_weight_decay_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="bm_weight_decay must be a finite non-negative"):
        _model().bm_loss(torch.randn(3, 4), bm_weight_decay=value)


def test_valid_weight_decay_still_returns_a_finite_loss() -> None:
    model = _model()
    base = model.bm_loss(torch.randn(3, 4))
    regularized = model.bm_loss(torch.randn(3, 4), bm_weight_decay=0.5)
    assert torch.isfinite(base) and torch.isfinite(regularized)
