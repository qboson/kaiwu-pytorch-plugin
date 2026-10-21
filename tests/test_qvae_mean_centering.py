"""Pin the dataset-mean centering applied before the QVAE encoder."""

from __future__ import annotations

import types

import numpy as np
import torch

from kaiwu.torch_plugin import BoltzmannMachine, QVAE


class _RecordingEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, inputs):
        self.last_input = inputs.detach().clone()
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


def _model(loss_type: str):
    encoder = _RecordingEncoder()
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
        bm=BoltzmannMachine(4),
        sampler=_Sampler(),
    )
    return model, encoder


def test_bernoulli_encoder_receives_raw_input_without_mean() -> None:
    model, encoder = _model("bernoulli")
    x = torch.rand(2, 6)
    model(x)
    torch.testing.assert_close(encoder.last_input, x)


def test_bernoulli_encoder_receives_centered_input() -> None:
    model, encoder = _model("bernoulli")
    model.set_dataset_mean(torch.full((6,), 0.25))
    x = torch.rand(2, 6)
    model(x)
    torch.testing.assert_close(encoder.last_input, x - 0.25)


def test_mse_encoder_does_not_center_even_with_a_mean() -> None:
    model, encoder = _model("mse")
    model.set_dataset_mean(torch.full((6,), 0.25))
    x = torch.rand(2, 6)
    model(x)
    torch.testing.assert_close(encoder.last_input, x)
