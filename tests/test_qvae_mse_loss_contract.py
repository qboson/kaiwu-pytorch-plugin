"""Pin the MSE reconstruction term of QVAE.loss."""

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
    return _QVAE(
        6,
        None,
        types.SimpleNamespace(
            num_latent_units=4,
            loss_type="mse",
            dist_beta=1.0,
            kl_beta=0.0,
            weight_decay=0.0,
        ),
        encoder=_Encoder(),
        decoder=_Decoder(),
        bm=BoltzmannMachine(4),
        sampler=_Sampler(),
    )


def test_mse_reconstruction_loss_matches_the_formula() -> None:
    model = _model()
    inputs = torch.rand(3, 6)
    recon, posterior, _, _ = model(inputs)

    loss = model.loss(inputs, recon, posterior)

    expected = torch.nn.functional.mse_loss(
        recon, inputs.view(-1, 6), reduction="sum"
    ) / inputs.size(0)
    torch.testing.assert_close(loss, expected)
