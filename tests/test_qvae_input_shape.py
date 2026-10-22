"""QVAE forward/energy must receive (batch, input_dimension) tensors."""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import QVAE, RestrictedBoltzmannMachine

INPUT_DIM = 6
LATENT_DIM = 4


class _Encoder(torch.nn.Module):  # pylint: disable=too-few-public-methods
    def forward(self, inputs):
        return torch.zeros(inputs.size(0), LATENT_DIM)


class _Decoder(torch.nn.Module):  # pylint: disable=too-few-public-methods
    def forward(self, zeta):
        return torch.zeros(zeta.size(0), INPUT_DIM)


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
        INPUT_DIM,
        None,
        types.SimpleNamespace(
            num_latent_units=LATENT_DIM,
            loss_type="mse",
            dist_beta=1.0,
            kl_beta=1.0,
            weight_decay=0.0,
        ),
        encoder=_Encoder(),
        decoder=_Decoder(),
        bm=RestrictedBoltzmannMachine(2, 2),
        sampler=_Sampler(),
    )


@pytest.mark.parametrize("shape", [(2, 3, INPUT_DIM), (INPUT_DIM,), (2, 1, INPUT_DIM)])
def test_forward_rejects_non_two_dimensional_inputs(shape) -> None:
    with pytest.raises(ValueError, match="x must have shape"):
        _model()(torch.rand(*shape))


@pytest.mark.parametrize("width", [INPUT_DIM - 1, INPUT_DIM + 1])
def test_forward_rejects_wrong_feature_width(width: int) -> None:
    with pytest.raises(ValueError, match="x must have shape"):
        _model()(torch.rand(2, width))


def test_energy_rejects_three_dimensional_input() -> None:
    with pytest.raises(ValueError, match="x must have shape"):
        _model().energy(torch.rand(2, 3, INPUT_DIM))


def test_two_dimensional_inputs_still_work() -> None:
    model = _model()
    inputs = torch.rand(3, INPUT_DIM)
    recon, posterior, q, zeta = model(inputs)
    assert recon.shape == (3, INPUT_DIM)
    assert q.shape == (3, LATENT_DIM)
    assert zeta.shape == (3, LATENT_DIM)
    assert posterior.logit_mu.shape == (3, LATENT_DIM)
    assert model.energy(inputs).shape == (3,)
