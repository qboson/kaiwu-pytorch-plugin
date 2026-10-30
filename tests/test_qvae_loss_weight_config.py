# -*- coding: utf-8 -*-
"""QVAE loss weights must be finite and non-negative."""

from __future__ import annotations

import math
from types import SimpleNamespace

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


def _model(**overrides) -> _DummyQVAE:
    config = SimpleNamespace(
        num_latent_units=4,
        loss_type="bernoulli",
        dist_beta=1.0,
        kl_beta=1e-6,
        weight_decay=0.0,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return _DummyQVAE(
        input_dimension=8,
        activation_fct=None,
        config=config,
        encoder=_Encoder(),
        decoder=_Decoder(),
        bm=RestrictedBoltzmannMachine(2, 2),
        sampler=_Sampler(),
    )


@pytest.mark.parametrize("field", ["kl_beta", "weight_decay"])
@pytest.mark.parametrize("value", [-1e-6, float("nan"), float("inf")])
def test_invalid_loss_weight_is_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        _model(**{field: value})


def test_zero_loss_weights_are_allowed() -> None:
    """A zero KL or weight-decay coefficient is a valid ablation."""
    model = _model(kl_beta=0.0, weight_decay=0.0)

    assert model.kl_beta == 0.0
    assert model.weight_decay == 0.0
    assert math.isfinite(model.kl_beta)
