"""QVAE.posterior must validate its latent-logits input."""

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


def test_one_dimensional_logits_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"q_logits must have shape \(batch, 4\)"):
        _model().posterior(torch.zeros(4), 1.0)


def test_wrong_latent_width_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"q_logits must have shape \(batch, 4\)"):
        _model().posterior(torch.zeros(2, 5), 1.0)


def test_three_dimensional_logits_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"q_logits must have shape \(batch, 4\)"):
        _model().posterior(torch.zeros(1, 2, 4), 1.0)


def test_valid_logits_still_sample_the_posterior() -> None:
    posterior, zeta = _model().posterior(torch.zeros(2, 4), 1.0)
    assert zeta.shape == (2, 4)
    assert posterior.smoothing_dist.beta > 0
