"""QVAE.bm_loss must validate its latent-state input."""

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


def test_empty_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="q must contain at least one sample"):
        _model().bm_loss(torch.zeros(0, 4))


def test_one_dimensional_state_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"q must have shape \(batch, 4\)"):
        _model().bm_loss(torch.zeros(4))


def test_wrong_latent_width_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"q must have shape \(batch, 4\)"):
        _model().bm_loss(torch.zeros(2, 5))


def test_valid_latent_state_returns_a_finite_loss() -> None:
    loss = _model().bm_loss(torch.randn(3, 4))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
