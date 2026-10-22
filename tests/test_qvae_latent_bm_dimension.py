"""Regression test for QVAE latent/BM dimension validation.

``QVAE.loss`` computes the cross-entropy term by feeding the encoder output
(width ``num_latent_units``) directly into the Boltzmann machine.  When the
supplied ``bm`` has a different number of nodes the failure surfaced deep
inside the RBM as an opaque shape error:

    RuntimeError: size mismatch, got input (2), mat (2x8), vec (4)

The pre-refactor implementation validated this and raised a descriptive
``ValueError``; the check must not be lost.
"""

import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import torch  # noqa: E402

from kaiwu.torch_plugin import RestrictedBoltzmannMachine  # noqa: E402
from kaiwu.torch_plugin.qvae import QVAE  # noqa: E402


class _Encoder(torch.nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.latent = latent

    def forward(self, x):
        return torch.ones(x.size(0), self.latent)


class _Decoder(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, z):
        return torch.zeros(z.size(0), self.input_dim)


class _Sampler:
    def solve(self, ising_matrix):
        return np.zeros((2, ising_matrix.shape[0]), dtype=np.float32)


class _QVAE(QVAE):
    def _create_encoder(self):
        return self.encoder

    def _create_decoder(self):
        return self.decoder

    def _create_bm(self):
        return self.bm

    def _create_sampler(self, sampler_type):
        del sampler_type
        return self.sampler


def _config(latent):
    return types.SimpleNamespace(
        num_latent_units=latent,
        loss_type="bernoulli",
        dist_beta=1.0,
        kl_beta=1e-6,
        weight_decay=0.0,
    )


def _build(latent, bm_visible, bm_hidden):
    return _QVAE(
        input_dimension=4,
        activation_fct=None,
        config=_config(latent),
        encoder=_Encoder(latent),
        decoder=_Decoder(4),
        bm=RestrictedBoltzmannMachine(bm_visible, bm_hidden),
        sampler=_Sampler(),
    )


def test_mismatched_bm_size_is_rejected_early():
    with pytest.raises(ValueError) as excinfo:
        _build(latent=8, bm_visible=2, bm_hidden=2)

    message = str(excinfo.value)
    assert "4" in message and "8" in message
    assert "Boltzmann machine" in message


def test_matching_bm_size_still_works():
    model = _build(latent=4, bm_visible=2, bm_hidden=2)
    x = torch.zeros(2, 4)
    recon_x, posterior, q, _ = model(x)
    loss = model.loss(x, recon_x, posterior)
    assert torch.isfinite(loss).item()
