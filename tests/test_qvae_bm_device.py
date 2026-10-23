"""Regression tests for QVAE / Boltzmann-machine device alignment.

The encoder, decoder, and BM are sibling submodules. Moving only ``qvae.bm``
across devices (as in two-phase BM fine-tuning) must not leave latent tensors
on the encoder device while BM parameters were relocated.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

import torch

tests_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(tests_dir, "../src")))
sys.path.insert(0, tests_dir)

from kaiwu.torch_plugin import RestrictedBoltzmannMachine

from test_qvae import DummyDecoder, DummyEncoder, DummyQVAE, DummySampler


class TestQVaeBmDevice(unittest.TestCase):
    """QVAE BM paths must follow the BM parameter device."""

    def setUp(self) -> None:
        self.input_dim = 8
        self.latent_dim = 4
        self.num_visible = self.latent_dim // 2
        self.num_hidden = self.latent_dim - self.num_visible
        self.config = types.SimpleNamespace(
            num_latent_units=self.latent_dim,
            loss_type="mse",
            dist_beta=1.0,
            kl_beta=1e-6,
            weight_decay=0.01,
        )
        self.encoder = DummyEncoder(self.latent_dim)
        self.decoder = DummyDecoder(self.input_dim)
        self.rbm = RestrictedBoltzmannMachine(self.num_visible, self.num_hidden)
        self.sampler = DummySampler()
        self.qvae = DummyQVAE(
            input_dimension=self.input_dim,
            activation_fct=None,
            config=self.config,
            encoder=self.encoder,
            decoder=self.decoder,
            bm=self.rbm,
            sampler=self.sampler,
        )

    def test_energy_after_bm_only_move(self) -> None:
        """``energy()`` must accept encoder outputs when only the BM moved."""
        self.qvae.bm.to(torch.device("meta"))
        x = torch.randn(2, self.input_dim)
        energy = self.qvae.energy(x)
        self.assertEqual(energy.device.type, "meta")

    def test_bm_loss_after_bm_only_move(self) -> None:
        """``bm_loss()`` must align detached logits with the BM device."""
        self.qvae.bm.to(torch.device("meta"))
        q = torch.ones(2, self.latent_dim)
        negative = torch.zeros(2, self.latent_dim, device=torch.device("meta"))
        with mock.patch.object(self.qvae.bm, "sample", return_value=negative):
            loss = self.qvae.bm_loss(q)
        self.assertEqual(loss.device.type, "meta")

    def test_kl_cross_entropy_after_bm_only_move(self) -> None:
        """KL cross-entropy must score BM energies on the BM parameter device."""
        self.qvae.bm.to(torch.device("meta"))
        logit_q = torch.zeros(2, self.latent_dim)
        negative = torch.zeros(2, self.latent_dim, device=torch.device("meta"))
        with mock.patch.object(self.qvae.bm, "sample", return_value=negative):
            cross_entropy = self.qvae._cross_entropy(logit_q)
        self.assertEqual(cross_entropy.device.type, "meta")


if __name__ == "__main__":
    unittest.main()
