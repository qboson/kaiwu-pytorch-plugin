# -*- coding: utf-8 -*-
"""Tests for the concrete QVAE example models."""

import os
import sys
import unittest
from types import SimpleNamespace

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SOURCE_ROOT = os.path.join(PROJECT_ROOT, "src")
MNIST_EXAMPLE_ROOT = os.path.join(PROJECT_ROOT, "example", "qvae_mnist")
CELL_EXAMPLE_ROOT = os.path.join(PROJECT_ROOT, "example", "qvae_cell")
sys.path.insert(0, SOURCE_ROOT)

import kaiwu

kaiwu.__path__.insert(0, os.path.join(SOURCE_ROOT, "kaiwu"))
for module_name in list(sys.modules):
    if module_name == "kaiwu.torch_plugin" or module_name.startswith(
        "kaiwu.torch_plugin."
    ):
        del sys.modules[module_name]

sys.path.insert(0, MNIST_EXAMPLE_ROOT)
sys.path.insert(0, CELL_EXAMPLE_ROOT)

from model import Config, MNISTQVAE
from models import CellQVAE, QVAEDecoder, QVAEEncoder


class DummyBoltzmannMachine(torch.nn.Module):
    """Minimal Boltzmann machine for example tests."""

    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.quadratic_coef = torch.nn.Parameter(
            torch.zeros(num_nodes // 2, num_nodes // 2)
        )
        self.linear_bias = torch.nn.Parameter(torch.zeros(num_nodes))

    def forward(self, states):
        """Return one energy value per state."""
        return -(states * self.linear_bias).sum(dim=-1)

    def sample(self, sampler):
        """Return deterministic negative samples."""
        del sampler
        return torch.zeros(2, self.num_nodes, device=self.linear_bias.device)

    def objective(self, positive, negative):
        """Return the contrastive energy objective."""
        return self(positive).mean() - self(negative).mean()


class TestExampleQVAEModels(unittest.TestCase):
    """Exercise the MNIST and single-cell QVAE implementations."""

    def test_mnist_qvae_forward_and_backward(self):
        config = Config(
            num_latent_units=8,
            encoder_hidden_nodes=[16],
            decoder_hidden_nodes=[16],
        )
        model = MNISTQVAE(
            input_dimension=8,
            activation_fct=torch.nn.ReLU(),
            config=config,
        )
        model.bm = DummyBoltzmannMachine(8)
        model.sampler = object()
        inputs = torch.rand(4, 8)

        reconstruction, posterior, logits, latent = model(inputs)
        loss = model.loss(inputs, reconstruction, posterior)
        loss.backward()

        self.assertEqual(reconstruction.shape, inputs.shape)
        self.assertEqual(logits.shape, (4, 8))
        self.assertEqual(latent.shape, (4, 8))
        self.assertTrue(torch.isfinite(loss))

    def test_mnist_qvae_energy(self):
        config = Config(
            num_latent_units=8,
            encoder_hidden_nodes=[16],
            decoder_hidden_nodes=[16],
        )
        config.loss_type = "mse"
        model = MNISTQVAE(
            input_dimension=8,
            activation_fct=torch.nn.ReLU(),
            config=config,
        )
        model.bm = DummyBoltzmannMachine(8)
        inputs = torch.rand(4, 8)

        energy = model.energy(inputs, "mse")

        self.assertEqual(energy.shape, (4,))
        self.assertTrue(torch.isfinite(energy).all())

    def test_cell_qvae_uses_standard_loss_and_bm_loss(self):
        config = SimpleNamespace(
            num_latent_units=8,
            loss_type="mse",
            dist_beta=1.0,
            kl_beta=1e-3,
            weight_decay=0.0,
        )
        model = CellQVAE(
            input_dimension=12,
            activation_fct=torch.nn.ReLU(),
            config=config,
            encoder=QVAEEncoder(12, 16, 8),
            decoder=QVAEDecoder(10, 16, 12),
            bm=DummyBoltzmannMachine(8),
            sampler=object(),
            n_batches=2,
        )
        inputs = torch.randn(4, 12)
        batch_indices = torch.tensor([0, 1, 0, 1])

        reconstruction, posterior, logits, latent = model(inputs, batch_indices)
        loss = model.loss(inputs, reconstruction, posterior)
        loss.backward()
        bm_loss = model.bm_loss(logits.detach())
        bm_loss.backward()

        expected_loss = model.last_recon_loss + config.kl_beta * model.last_kl_loss
        self.assertEqual(reconstruction.shape, inputs.shape)
        self.assertEqual(logits.shape, (4, 8))
        self.assertEqual(latent.shape, (4, 8))
        self.assertTrue(torch.allclose(loss.detach(), expected_loss))
        self.assertTrue(torch.isfinite(bm_loss))


if __name__ == "__main__":
    unittest.main()
