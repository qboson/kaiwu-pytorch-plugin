# -*- coding: utf-8 -*-
"""QVAE Bernoulli train bias rejects non-finite dataset means."""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.torch_plugin.qvae import QVAE


class _Encoder(torch.nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, x):
        return torch.ones(x.size(0), self.latent_dim, dtype=x.dtype)


class _Decoder(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim

    def forward(self, z):
        return torch.zeros(z.size(0), self.input_dim, dtype=z.dtype)


class _Sampler:
    def solve(self, ising_matrix):
        return np.zeros((1, ising_matrix.shape[0]), dtype=np.float32)


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
        encoder=_Encoder(4),
        decoder=_Decoder(8),
        bm=RestrictedBoltzmannMachine(2, 2),
        sampler=_Sampler(),
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_scalar_mean_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite values"):
        _model().set_train_bias(value)


def test_nonfinite_vector_mean_is_rejected() -> None:
    mean = torch.full((8,), float("nan"))

    with pytest.raises(ValueError, match="finite values"):
        _model().set_train_bias(mean)


def test_finite_mean_still_sets_a_finite_bias() -> None:
    model = _model()

    model.set_train_bias(0.5)

    assert torch.isfinite(model._train_bias).all()
    assert torch.allclose(
        model._train_bias, torch.zeros_like(model._train_bias), atol=1e-6
    )
