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
        # BM 维度分解：可见单元数 V，隐藏单元数 H，必须满足 V + H == L
        self.num_visible = self.latent_dim // 2  # V = 2
        self.num_hidden = self.latent_dim - self.num_visible  # H = 2

        self.config = types.SimpleNamespace(
            num_latent_units=self.latent_dim,
            loss_type="bernoulli",
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
        self.assertEqual(q.shape, (x.size(0), self.latent_dim))
        self.assertEqual(zeta.shape, (x.size(0), self.latent_dim))
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

        self.assertEqual(energy.shape, (x.size(0),))
        torch.testing.assert_close(self.encoder.inputs[-1], x - 0.5)
        torch.testing.assert_close(self.qvae.energy(x, loss_type="bernoulli"), energy)
        with self.assertRaises(ValueError):
            self.qvae.energy(x, loss_type="mse")

    def test_energy_accepts_bernoulli_without_dataset_mean(self):
        """Bernoulli energy uses raw inputs when no dataset mean is supplied."""
        self.qvae.eval()
        x = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4) / 16
        _, _, q, _ = self.qvae(x)
        expected_energy = self.rbm((q > 0).float())

        for loss_type in (None, "bernoulli"):
            with self.subTest(loss_type=loss_type):
                energy = self.qvae.energy(x, loss_type=loss_type)
                self.assertEqual(energy.shape, (x.size(0),))
                torch.testing.assert_close(self.encoder.inputs[-1], x.reshape(2, -1))
                torch.testing.assert_close(energy, expected_energy)

    def test_bernoulli_energy_with_real_encoder_and_bm_gradients(self):
        """Uncentered energy scores distinct states and trains only the BM."""
        encoder = torch.nn.Linear(self.input_dim, self.latent_dim)
        with torch.no_grad():
            encoder.weight.zero_()
            encoder.weight[:, : self.latent_dim].copy_(torch.eye(self.latent_dim))
            encoder.bias.copy_(torch.tensor([-0.5, 0.25, -1.0, 0.0]))
        bm = RestrictedBoltzmannMachine(
            2,
            2,
            quadratic_coef=torch.tensor([[0.5, -1.0], [2.0, 0.25]]),
            linear_bias=torch.tensor([0.1, -0.2, 0.3, 0.4]),
            device="cpu",
        )
        self.qvae.encoder = encoder
        self.qvae.bm = bm
        x = torch.tensor(
            [
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        # Thresholded encoder states are [0, 1, 0, 0] and [1, 1, 0, 1].
        energy = self.qvae.energy(x)
        torch.testing.assert_close(energy, torch.tensor([0.2, 0.45]))
        energy.sum().backward()
        torch.testing.assert_close(
            bm.linear_bias.grad, torch.tensor([-1.0, -2.0, 0.0, -1.0])
        )
        torch.testing.assert_close(
            bm.quadratic_coef.grad, torch.tensor([[0.0, -1.0], [0.0, -1.0]])
        )
        self.assertIsNone(encoder.weight.grad)
        self.assertIsNone(encoder.bias.grad)

        # Explicit encoder/BM components can use different precisions, and a
        # converted double model must score its generated binary states too.
        for encoder_dtype, bm_dtype in (
            (torch.float32, torch.float64),
            (torch.float64, torch.float64),
            (torch.float16, torch.float32),
        ):
            with self.subTest(encoder_dtype=encoder_dtype, bm_dtype=bm_dtype):
                encoder.to(dtype=encoder_dtype)
                bm.double() if bm_dtype == torch.float64 else bm.float()
                bm.zero_grad()
                typed_energy = self.qvae.energy(x.to(dtype=encoder_dtype))
                torch.testing.assert_close(
                    typed_energy, torch.tensor([0.2, 0.45], dtype=bm_dtype)
                )
                typed_energy.sum().backward()
                torch.testing.assert_close(
                    bm.linear_bias.grad,
                    torch.tensor([-1.0, -2.0, 0.0, -1.0], dtype=bm_dtype),
                )
                self.assertIsNone(encoder.weight.grad)

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

        energy = self.qvae.energy(x)
        torch.testing.assert_close(self.encoder.inputs[-1], x)
        torch.testing.assert_close(energy, self.rbm(torch.ones(2, self.latent_dim)))

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
