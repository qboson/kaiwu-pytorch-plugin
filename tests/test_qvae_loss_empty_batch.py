"""QVAE.loss must reject an empty batch instead of returning NaN."""

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
            loss_type="mse",
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
    model = _model()
    inputs = torch.zeros(0, 6)
    recon, posterior, _, _ = model(inputs)
    with pytest.raises(ValueError, match="x must contain at least one sample"):
        model.loss(inputs, recon, posterior)


def test_non_empty_batch_still_returns_a_finite_loss() -> None:
    model = _model()
    inputs = torch.rand(3, 6)
    recon, posterior, _, _ = model(inputs)
    loss = model.loss(inputs, recon, posterior)
    assert torch.isfinite(loss)
