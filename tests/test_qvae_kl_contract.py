"""Pin the QVAE KL term computed from the posterior."""

from __future__ import annotations

import types

import numpy as np
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
    torch.manual_seed(0)
    bm = BoltzmannMachine(4)
    with torch.no_grad():
        bm.quadratic_coef.copy_(torch.randn_like(bm.quadratic_coef))
        bm.linear_bias.copy_(torch.randn_like(bm.linear_bias))
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
        bm=bm,
        sampler=_Sampler(),
    )


def test_kl_term_is_cross_entropy_minus_entropy() -> None:
    model = _model()
    q = torch.randn(3, 4)
    posterior, _ = model.posterior(q, 1.0)

    kl = model._kl_dist_from(posterior)  # pylint: disable=protected-access

    expected = model._cross_entropy(q) - posterior.entropy().sum(dim=1)  # pylint: disable=protected-access
    assert kl.shape == (3,)
    torch.testing.assert_close(kl, expected)
