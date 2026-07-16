"""QVAE public interface and loss behavior tests."""

import types
import unittest

import numpy as np
import torch

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.torch_plugin.qvae import QVAE


class DummyEncoder(torch.nn.Module):
    """Return deterministic latent logits for QVAE tests."""

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return torch.ones(x.size(0), self.latent_dim, device=x.device, dtype=x.dtype)


class DummyDecoder(torch.nn.Module):
    """Return zero reconstruction logits with the requested input dimension."""

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, z):
        return torch.zeros(z.size(0), self.input_dim, device=z.device, dtype=z.dtype)


class DummySampler:
    """Return a deterministic set of binary BM samples."""

    def solve(self, ising_matrix):
        return np.zeros((2, ising_matrix.shape[0]), dtype=np.float32)


class DummyQVAE(QVAE):
    """QVAE test subclass using the supplied components."""

    def _create_encoder(self):
        return self.encoder

    def _create_decoder(self):
        return self.decoder

    def _create_bm(self):
        return self.bm

    def _create_sampler(self, sampler_type):
        del sampler_type
        return self.sampler


class TestQVAE(unittest.TestCase):
    """Test QVAE construction, forward pass, energy, and losses."""

    def setUp(self):
        self.input_dim = 8
        self.latent_dim = 4
        self.config = types.SimpleNamespace(
            num_latent_units=self.latent_dim,
            loss_type="bernoulli",
            dist_beta=1.0,
            kl_beta=1e-6,
            weight_decay=0.01,
        )
        self.encoder = DummyEncoder(self.latent_dim)
        self.decoder = DummyDecoder(self.input_dim)
        self.rbm = RestrictedBoltzmannMachine(2, 2)
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

    def test_constructor_reuses_supplied_components(self):
        """Explicit components take precedence over factory methods."""
        self.assertIs(self.qvae.encoder, self.encoder)
        self.assertIs(self.qvae.decoder, self.decoder)
        self.assertIs(self.qvae.bm, self.rbm)
        self.assertIs(self.qvae.sampler, self.sampler)
        self.assertEqual(self.qvae.sampler_type, "sa")

    def test_forward_adds_dataset_bias(self):
        """Bernoulli forward output includes the bias derived from the mean."""
        self.qvae.eval()
        self.qvae.set_dataset_mean(torch.full((self.input_dim,), 0.25))
        self.qvae.set_train_bias(torch.full((self.input_dim,), 0.25))
        x = torch.zeros(2, self.input_dim)

        recon_x, posterior, q, zeta = self.qvae(x)

        expected_bias = torch.full((self.input_dim,), -torch.log(torch.tensor(3.0)))
        torch.testing.assert_close(recon_x, expected_bias.expand_as(recon_x))
        self.assertEqual(q.shape, (2, self.latent_dim))
        self.assertEqual(zeta.shape, (2, self.latent_dim))
        self.assertEqual(posterior.logit_mu.shape, q.shape)
        self.assertIn("_train_bias", dict(self.qvae.named_buffers()))

    def test_set_train_bias_accepts_scalar_and_rejects_wrong_shape(self):
        """Train bias accepts a scalar mean and validates vector dimensions."""
        self.qvae.set_train_bias(0.5)
        torch.testing.assert_close(self.qvae._train_bias, torch.zeros(self.input_dim))

        with self.assertRaises(ValueError):
            self.qvae.set_train_bias(torch.zeros(self.input_dim - 1))

    def test_energy_uses_centered_input_and_optional_loss_type(self):
        """Energy uses the configured loss type and centers Bernoulli inputs."""
        self.qvae.set_dataset_mean(torch.full((self.input_dim,), 0.5))
        x = torch.ones(2, self.input_dim)
        energy = self.qvae.energy(x)

        self.assertEqual(energy.shape, (2,))
        torch.testing.assert_close(self.encoder.inputs[-1], x - 0.5)
        torch.testing.assert_close(self.qvae.energy(x, loss_type="bernoulli"), energy)
        with self.assertRaises(ValueError):
            self.qvae.energy(x, loss_type="mse")

    def test_mse_forward_and_loss(self):
        """The MSE configuration bypasses Bernoulli centering and bias."""
        self.config.loss_type = "mse"
        self.qvae.set_dataset_mean(torch.full((self.input_dim,), 0.25))
        self.qvae.eval()
        x = torch.ones(2, self.input_dim)
        recon_x, posterior, _, _ = self.qvae(x)

        self.assertTrue(torch.equal(recon_x, torch.zeros_like(recon_x)))
        torch.testing.assert_close(self.encoder.inputs[-1], x)
        self.assertGreater(self.qvae.loss(x, recon_x, posterior).item(), 0.0)

    def test_unsupported_loss_type_is_rejected(self):
        """Unsupported loss types fail at the public computation boundary."""
        self.config.loss_type = "unsupported"
        x = torch.zeros(2, self.input_dim)
        with self.assertRaises(ValueError):
            self.qvae(x)
        with self.assertRaises(ValueError):
            self.qvae.energy(x)


if __name__ == "__main__":
    unittest.main()
