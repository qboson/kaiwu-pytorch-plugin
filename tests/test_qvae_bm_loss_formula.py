"""Pin the QVAE BM-loss formula and its weight-decay term."""

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


def test_bm_loss_matches_the_objective_and_decay() -> None:
    model = _model()
    q = torch.randn(3, 4)

    negative = model.bm.sample(model.sampler)
    expected = model.bm.objective(torch.sigmoid(q.detach()), negative)
    torch.testing.assert_close(model.bm_loss(q), expected)

    decay = 0.7
    expected += decay * torch.sum(model.bm.quadratic_coef**2)
    expected += decay * 0.5 * torch.sum(model.bm.linear_bias**2)
    torch.testing.assert_close(model.bm_loss(q, bm_weight_decay=decay), expected)


def test_bm_loss_does_not_detach_gradients_through_decay() -> None:
    model = _model()
    q = torch.randn(2, 4)
    loss = model.bm_loss(q, bm_weight_decay=0.1)
    loss.backward()
    assert model.bm.quadratic_coef.grad is not None
