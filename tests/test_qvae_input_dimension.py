"""QVAE must validate input_dimension at construction."""

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


def _build(input_dimension) -> _QVAE:
    return _QVAE(
        input_dimension,
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


@pytest.mark.parametrize("value", [0, -3, [-2]])
def test_non_positive_input_dimension_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="input_dimension must be a positive integer"):
        _build(value)


@pytest.mark.parametrize("value", [6.0, True, "6", None, [6.0]])
def test_non_integer_input_dimension_is_rejected(value) -> None:
    with pytest.raises(TypeError, match="input_dimension must be a positive integer"):
        _build(value)


def test_empty_dimension_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="input_dimension must be a positive integer"):
        _build([])


def test_valid_dimensions_still_build_the_model() -> None:
    model = _build(6)
    assert model._input_dimension == 6  # pylint: disable=protected-access
    assert model.forward(torch.rand(2, 6))[0].shape == (2, 6)

    assert _build([6])._input_dimension == 6  # pylint: disable=protected-access
