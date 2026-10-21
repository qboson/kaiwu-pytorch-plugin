"""Pin the Bernoulli train-bias formula and its clipping."""

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


def _expected_bias(mean: float) -> torch.Tensor:
    clipped = min(max(mean, 0.001), 0.999)
    return torch.full((6,), -np.log(1.0 / clipped - 1.0), dtype=torch.float32)


def test_bias_matches_the_logit_of_the_mean() -> None:
    model = _model()
    model.set_train_bias(0.25)
    torch.testing.assert_close(model._train_bias, _expected_bias(0.25))  # pylint: disable=protected-access


def test_bias_clips_zero_and_one_means() -> None:
    model = _model()
    model.set_train_bias(0.0)
    torch.testing.assert_close(model._train_bias, _expected_bias(0.0))  # pylint: disable=protected-access
    assert torch.isfinite(model._train_bias).all()  # pylint: disable=protected-access

    model.set_train_bias(1.0)
    torch.testing.assert_close(model._train_bias, _expected_bias(1.0))  # pylint: disable=protected-access
    assert torch.isfinite(model._train_bias).all()  # pylint: disable=protected-access


def test_vector_mean_and_scalar_mean_agree() -> None:
    scalar_model = _model()
    scalar_model.set_train_bias(0.25)
    vector_model = _model()
    vector_model.set_train_bias(torch.full((6,), 0.25))
    torch.testing.assert_close(  # pylint: disable=protected-access
        scalar_model._train_bias, vector_model._train_bias  # pylint: disable=protected-access
    )
