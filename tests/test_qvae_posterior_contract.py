"""Pin the object returned by QVAE.posterior."""

from __future__ import annotations

import types

import numpy as np
import torch

from kaiwu.torch_plugin import BoltzmannMachine, QVAE
from kaiwu.torch_plugin.qvae_dist_util import MixtureGeneric


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


def test_posterior_wraps_the_logits_with_the_requested_smoothing() -> None:
    model = _model()
    logits = torch.randn(3, 4)

    posterior, zeta = model.posterior(logits, 1.5)

    assert isinstance(posterior, MixtureGeneric)
    assert posterior.logit_mu is logits
    assert zeta.shape == logits.shape
    torch.testing.assert_close(
        posterior.smoothing_dist.beta, torch.tensor(1.5)
    )
