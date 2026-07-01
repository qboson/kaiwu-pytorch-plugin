# -*- coding: utf-8 -*-
"""Unit tests for the refactored QVAE base implementation."""

import os
import sys
import unittest
from types import SimpleNamespace

import torch

SOURCE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, SOURCE_ROOT)

# The kaiwu core package is installed separately, so extend its package path to
# load torch_plugin from this checkout instead of the installed plugin version.
import kaiwu

kaiwu.__path__.insert(0, os.path.join(SOURCE_ROOT, "kaiwu"))
for module_name in list(sys.modules):
    if module_name == "kaiwu.torch_plugin" or module_name.startswith(
        "kaiwu.torch_plugin."
    ):
        del sys.modules[module_name]

from kaiwu.torch_plugin.qvae import QVAE


class DummyBoltzmannMachine(torch.nn.Module):
    """Minimal differentiable Boltzmann machine used by QVAE tests."""

    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.quadratic_coef = torch.nn.Parameter(torch.zeros(num_nodes // 2, num_nodes // 2))
        self.linear_bias = torch.nn.Parameter(torch.zeros(num_nodes))

    def forward(self, states):
        """Return a simple energy for each state."""
        return -(states * self.linear_bias).sum(dim=-1)

    def sample(self, sampler):
        """Return deterministic negative samples."""
        del sampler
        return torch.zeros(2, self.num_nodes, device=self.linear_bias.device)

    def objective(self, positive, negative):
        """Return a contrastive objective connected to BM parameters."""
        return self(positive).mean() - self(negative).mean()


class ConcreteQVAE(QVAE):
    """Small concrete QVAE implementation for unit tests."""

    def _create_encoder(self):
        return torch.nn.Linear(self._input_dimension, self._latent_dimensions)

    def _create_decoder(self):
        return torch.nn.Linear(self._latent_dimensions, self._input_dimension)

    def _create_bm(self):
        return DummyBoltzmannMachine(self._latent_dimensions)

    def _create_sampler(self, sampler_type):
        return sampler_type


class TestQVAE(unittest.TestCase):
    """Verify the current QVAE API and supported reconstruction losses."""

    def _create_model(self, loss_type="bernoulli"):
        config = SimpleNamespace(
            num_latent_units=8,
            loss_type=loss_type,
            dist_beta=1.0,
            kl_beta=1e-3,
            weight_decay=1e-2,
        )
        model = ConcreteQVAE(
            input_dimension=8,
            activation_fct=torch.nn.ReLU(),
            config=config,
        )
        model.create_networks()
        return model

    def test_bernoulli_forward_without_dataset_mean(self):
        model = self._create_model()
        inputs = torch.rand(2, 8)

        reconstruction, posterior, logits, latent = model(inputs)
        loss = model.loss(inputs, reconstruction, posterior)

        self.assertEqual(reconstruction.shape, inputs.shape)
        self.assertEqual(logits.shape, inputs.shape)
        self.assertEqual(latent.shape, inputs.shape)
        self.assertEqual(loss.ndim, 0)

    def test_mse_forward_and_backward(self):
        model = self._create_model(loss_type="mse")
        inputs = torch.randn(2, 8)

        reconstruction, posterior, _, _ = model(inputs)
        loss = model.loss(inputs, reconstruction, posterior)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.encoder.weight.grad)

    def test_train_bias_is_persistent_buffer(self):
        model = self._create_model()
        model.set_dataset_mean(torch.full((8,), 0.25))
        model.set_train_bias(torch.full((8,), 0.25))

        self.assertIn("_train_bias", dict(model.named_buffers()))
        self.assertIn("_train_bias", model.state_dict())
        self.assertFalse(model._train_bias.requires_grad)

    def test_scalar_train_bias_is_expanded(self):
        model = self._create_model()
        model.set_train_bias(0.25)

        self.assertEqual(model._train_bias.shape, (8,))

    def test_explicit_components_are_preserved(self):
        config = SimpleNamespace(
            num_latent_units=8,
            loss_type="mse",
            dist_beta=1.0,
            kl_beta=1e-3,
            weight_decay=1e-2,
        )
        encoder = torch.nn.Linear(8, 8)
        decoder = torch.nn.Linear(8, 8)
        bm = DummyBoltzmannMachine(8)
        sampler = object()

        model = ConcreteQVAE(
            input_dimension=8,
            activation_fct=torch.nn.ReLU(),
            config=config,
            encoder=encoder,
            decoder=decoder,
            bm=bm,
            sampler=sampler,
        )

        self.assertIs(model.encoder, encoder)
        self.assertIs(model.decoder, decoder)
        self.assertIs(model.bm, bm)
        self.assertIs(model.sampler, sampler)


if __name__ == "__main__":
    unittest.main()
