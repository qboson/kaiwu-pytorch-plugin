"""QVAE.energy must work for the configured loss type without a dataset mean."""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine, QVAE

LOGITS = torch.tensor([[-2.0, 0.5, 3.0, -1.0]])


class _Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, inputs):
        self.last_input = inputs.detach().clone()
        return LOGITS.expand(inputs.size(0), -1)


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


def _model(loss_type: str = "bernoulli"):
    torch.manual_seed(0)
    bm = BoltzmannMachine(4)
    with torch.no_grad():
        bm.quadratic_coef.copy_(torch.randn_like(bm.quadratic_coef))
        bm.linear_bias.copy_(torch.randn_like(bm.linear_bias))
    encoder = _Encoder()
    model = _QVAE(
        6,
        None,
        types.SimpleNamespace(
            num_latent_units=4,
            loss_type=loss_type,
            dist_beta=1.0,
            kl_beta=1.0,
            weight_decay=0.0,
        ),
        encoder=encoder,
        decoder=_Decoder(),
        bm=bm,
        sampler=_Sampler(),
    )
    return model, encoder


def test_bernoulli_energy_works_without_a_dataset_mean() -> None:
    model, encoder = _model()
    x = torch.rand(2, 6)

    energy = model.energy(x)

    expected = model.bm((LOGITS.expand(2, 4) > 0).float())
    torch.testing.assert_close(energy, expected)
    assert torch.isfinite(energy).all()
    torch.testing.assert_close(encoder.last_input, x)


def test_bernoulli_energy_centers_the_encoder_input() -> None:
    model, encoder = _model()
    model.set_dataset_mean(torch.full((6,), 0.25))
    x = torch.rand(2, 6)
    model.energy(x)
    torch.testing.assert_close(encoder.last_input, x - 0.25)


def test_mse_energy_ignores_the_dataset_mean() -> None:
    model, encoder = _model("mse")
    model.set_dataset_mean(torch.full((6,), 0.25))
    x = torch.rand(2, 6)
    model.energy(x)
    torch.testing.assert_close(encoder.last_input, x)


def test_unknown_loss_type_is_still_rejected() -> None:
    model, _ = _model("bogus")
    with pytest.raises(ValueError, match="Unsupported loss type"):
        model.energy(torch.rand(2, 6))
