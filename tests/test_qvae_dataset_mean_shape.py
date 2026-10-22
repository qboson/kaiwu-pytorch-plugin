"""QVAE.set_dataset_mean must reject a mean that cannot represent the inputs."""

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


def test_wrong_width_mean_is_rejected() -> None:
    with pytest.raises(ValueError, match="Dataset mean must be a scalar or match"):
        _model().set_dataset_mean(torch.zeros(5))


def test_wrong_width_positional_mean_is_rejected() -> None:
    with pytest.raises(ValueError, match="Dataset mean must be a scalar or match"):
        _model().set_dataset_mean([[0.1, 0.2], [0.3, 0.4]])


def test_scalar_and_matching_mean_are_accepted() -> None:
    model = _model()
    model.set_dataset_mean(0.25)
    assert model.forward(torch.rand(2, 6))[0].shape == (2, 6)

    model.set_dataset_mean(torch.full((6,), 0.25))
    assert model.forward(torch.rand(2, 6))[0].shape == (2, 6)


def test_none_clears_the_mean() -> None:
    model = _model()
    model.set_dataset_mean(torch.full((6,), 0.25))
    model.set_dataset_mean(None)
    assert model._dataset_mean is None  # pylint: disable=protected-access
