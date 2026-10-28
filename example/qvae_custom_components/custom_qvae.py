# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Inject your own encoder, decoder, Boltzmann machine, and sampler into QVAE.

``QVAE`` builds missing components through ``_create_encoder`` /
``_create_decoder`` / ``_create_bm`` / ``_create_sampler``. Components passed to
the constructor take precedence, which lets a subclass reuse an existing
encoder, decoder, BM, or sampler instead of rebuilding it.

This example wires four toy components into a small subclass, runs a forward
pass and one loss backward, and checks that the encoder sees the mean-centered
input for the Bernoulli objective. It needs no dataset, checkpoint, or Kaiwu
license.
"""

from __future__ import annotations

import types

import numpy as np
import torch
from torch import nn

from kaiwu.torch_plugin import QVAE, RestrictedBoltzmannMachine

INPUT_DIM = 6
LATENT_DIM = 4
BATCH = 5


class ToyEncoder(nn.Module):  # pylint: disable=too-few-public-methods
    """Maps inputs to latent logits and records what it receives."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(INPUT_DIM, LATENT_DIM)
        self.seen = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Returns latent logits and records the encoder input."""
        self.seen.append(inputs.detach().clone())
        return self.linear(inputs)


class ToyDecoder(nn.Module):  # pylint: disable=too-few-public-methods
    """Maps latent samples back to reconstruction logits."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(LATENT_DIM, INPUT_DIM)

    def forward(self, zeta: torch.Tensor) -> torch.Tensor:
        """Returns reconstruction logits."""
        return self.linear(zeta)


class ConstantSampler:  # pylint: disable=too-few-public-methods
    """Deterministic sampler returning two opposite spin states."""

    def solve(self, ising_matrix: np.ndarray) -> np.ndarray:
        """Returns two ``-1``/``1`` spin vectors for the given Ising matrix."""
        spins = np.ones((2, ising_matrix.shape[0]), dtype=np.float32)
        spins[1] *= -1
        return spins


class InjectedQVAE(QVAE):  # pylint: disable=too-few-public-methods
    """QVAE subclass that reuses the components supplied to the constructor."""

    def _create_encoder(self):
        return self.encoder

    def _create_decoder(self):
        return self.decoder

    def _create_bm(self):
        return self.bm

    def _create_sampler(self, sampler_type):
        del sampler_type
        return self.sampler


def build_config() -> types.SimpleNamespace:
    """Builds the minimal configuration QVAE reads during construction."""
    return types.SimpleNamespace(
        num_latent_units=LATENT_DIM,
        loss_type="bernoulli",
        dist_beta=1.0,
        kl_beta=1e-3,
        weight_decay=0.0,
    )


def build_model() -> InjectedQVAE:
    """Builds an InjectedQVAE with toy components on the latent-width BM."""
    num_visible = LATENT_DIM // 2
    bm = RestrictedBoltzmannMachine(
        num_visible=num_visible, num_hidden=LATENT_DIM - num_visible
    )
    return InjectedQVAE(
        INPUT_DIM,
        nn.SiLU(),
        build_config(),
        encoder=ToyEncoder(),
        decoder=ToyDecoder(),
        bm=bm,
        sampler=ConstantSampler(),
    )


def main() -> None:
    """Runs the injection checks for one forward and one backward pass."""
    torch.manual_seed(0)
    model = build_model()

    assert model.encoder is not None and model.decoder is not None
    assert model.bm is not None and model.sampler is not None

    inputs = torch.rand(BATCH, INPUT_DIM)
    recon, posterior, q, zeta = model(inputs)
    print("recon:", tuple(recon.shape), "q:", tuple(q.shape), "zeta:", tuple(zeta.shape))
    assert recon.shape == (BATCH, INPUT_DIM)
    assert q.shape == (BATCH, LATENT_DIM) and zeta.shape == (BATCH, LATENT_DIM)

    model.set_dataset_mean(torch.full((INPUT_DIM,), 0.5))
    model.set_train_bias(torch.full((INPUT_DIM,), 0.5))
    encoder = model.encoder
    encoder.seen.clear()
    recon, posterior, _, _ = model(inputs)
    centered = encoder.seen[-1]
    print("encoder sees mean-centered input:", bool(torch.allclose(centered, inputs - 0.5)))
    assert torch.allclose(centered, inputs - 0.5)

    loss = model.loss(inputs, recon, posterior)
    loss.backward()
    grads = [parameter.grad is not None for parameter in model.encoder.parameters()]
    print("loss:", round(float(loss), 4), "encoder grads:", grads)
    assert torch.isfinite(loss)
    assert all(grads)

    energy = model.energy(inputs)
    print("energy shape:", tuple(energy.shape))
    assert energy.shape == (BATCH,)
    assert bool(torch.isfinite(energy).all())
    print("custom components OK")


if __name__ == "__main__":
    main()
