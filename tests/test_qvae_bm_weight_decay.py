# -*- coding: utf-8 -*-
"""QVAE BM weight decay must be finite and non-negative."""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.torch_plugin.qvae import QVAE


class _Encoder(torch.nn.Module):
    def forward(self, x):
        return torch.ones(x.size(0), 4, dtype=x.dtype)


class _Decoder(torch.nn.Module):
    def forward(self, z):
        return torch.zeros(z.size(0), 8, dtype=z.dtype)


class _Sampler:
    def solve(self, ising_matrix):
        return np.zeros((2, ising_matrix.shape[0]), dtype=np.float32)


class _DummyQVAE(QVAE):
    def _create_encoder(self):
        return self.encoder

    def _create_decoder(self):
        return self.decoder

    def _create_bm(self):
        return self.bm

    def _create_sampler(self, sampler_type):
        del sampler_type
        return self.sampler


def _model() -> _DummyQVAE:
    config = types.SimpleNamespace(
        num_latent_units=4,
        loss_type="bernoulli",
        dist_beta=1.0,
        kl_beta=1e-6,
        weight_decay=0.0,
    )
    return _DummyQVAE(
        input_dimension=8,
        activation_fct=None,
        config=config,
        encoder=_Encoder(),
        decoder=_Decoder(),
        bm=RestrictedBoltzmannMachine(2, 2),
        sampler=_Sampler(),
    )


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_invalid_bm_weight_decay_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        _model().bm_loss(torch.zeros(2, 4), bm_weight_decay=value)


def test_valid_bm_weight_decay_returns_a_finite_loss() -> None:
    loss = _model().bm_loss(torch.zeros(2, 4), bm_weight_decay=0.1)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
