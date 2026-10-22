"""Pin the Bernoulli reconstruction bias added by QVAE.forward."""

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


def _model(loss_type: str = "bernoulli") -> _QVAE:
    return _QVAE(
        6,
        None,
        types.SimpleNamespace(
            num_latent_units=4,
            loss_type=loss_type,
            dist_beta=1.0,
            kl_beta=1.0,
            weight_decay=0.0,
        ),
        encoder=_Encoder(),
        decoder=_Decoder(),
        bm=BoltzmannMachine(4),
        sampler=_Sampler(),
    )


def test_bernoulli_reconstruction_includes_the_train_bias() -> None:
    model = _model()
    inputs = torch.rand(3, 6)
    assert torch.allclose(model(inputs)[0], torch.zeros(3, 6))

    model.set_train_bias(0.25)
    expected = torch.full((3, 6), float(-np.log(1 / 0.25 - 1)))
    torch.testing.assert_close(model(inputs)[0], expected)


def test_mse_reconstruction_ignores_the_train_bias() -> None:
    model = _model("mse")
    model.set_train_bias(0.25)
    torch.testing.assert_close(model(torch.rand(2, 6))[0], torch.zeros(2, 6))
