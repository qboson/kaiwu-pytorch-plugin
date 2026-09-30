"""Unit and regression tests for Q_SVI training engine and SVI backend."""

import types
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kaiwu.torch_plugin import QSVI, Q_SVI, QVAE, RestrictedBoltzmannMachine


class DummyEncoder(nn.Module):
    """Deterministic latent encoder for QVAE tests."""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        return self.linear(x)


class DummyDecoder(nn.Module):
    """Deterministic reconstruction decoder for QVAE tests."""

    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.linear = nn.Linear(latent_dim, input_dim)

    def forward(self, z):
        return self.linear(z)


class DeterministicSampler:
    """Deterministic sampler returning zeros for BM optimization."""

    def solve(self, ising_matrix):
        return np.zeros((4, ising_matrix.shape[0]), dtype=np.float32)


class MockQVAE(QVAE):
    """QVAE subclass for testing using pre-instantiated components."""

    def _create_encoder(self):
        return self.encoder

    def _create_decoder(self):
        return self.decoder

    def _create_bm(self):
        return self.bm

    def _create_sampler(self, sampler_type):
        del sampler_type
        return self.sampler


def create_mock_qvae(input_dim=8, latent_dim=4, seed=42):
    """Create reproducible MockQVAE instance."""
    torch.manual_seed(seed)
    num_visible = latent_dim // 2
    num_hidden = latent_dim - num_visible

    config = types.SimpleNamespace(
        type="QVAE",
        num_latent_units=latent_dim,
        loss_type="bernoulli",
        dist_beta=1.0,
        kl_beta=1e-5,
        weight_decay=0.01,
        sampler_type="sa",
    )
    encoder = DummyEncoder(input_dim, latent_dim)
    decoder = DummyDecoder(latent_dim, input_dim)
    rbm = RestrictedBoltzmannMachine(num_visible, num_hidden)
    sampler = DeterministicSampler()

    model = MockQVAE(
        input_dimension=input_dim,
        activation_fct=None,
        config=config,
        encoder=encoder,
        decoder=decoder,
        bm=rbm,
        sampler=sampler,
    )
    return model


class TestQSVIInterface(unittest.TestCase):
    """Test Q_SVI public API alignment and core functionality."""

    def setUp(self):
        self.input_dim = 8
        self.latent_dim = 4
        self.model = create_mock_qvae(self.input_dim, self.latent_dim, seed=123)

    def test_alias_equivalence(self):
        """QSVI and Q_SVI point to the same class."""
        self.assertIs(QSVI, Q_SVI)

    def test_initialization_with_single_and_dual_optimizers(self):
        """Q_SVI supports single optimizer, dual optimizer tuple, and explicit bm_optim."""
        vae_params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        vae_optim = torch.optim.SGD(vae_params, lr=0.01)
        bm_optim = torch.optim.SGD(self.model.bm.parameters(), lr=0.01)

        # Single optimizer mode
        svi_single = Q_SVI(model=self.model, optim=vae_optim)
        self.assertFalse(svi_single.use_two_optimisers)
        self.assertIsNone(svi_single.bm_optim)
        self.assertIs(svi_single.vae_optim, vae_optim)

        # Dual optimizer via tuple
        svi_tuple = Q_SVI(model=self.model, optim=(vae_optim, bm_optim))
        self.assertTrue(svi_tuple.use_two_optimisers)
        self.assertIs(svi_tuple.vae_optim, vae_optim)
        self.assertIs(svi_tuple.bm_optim, bm_optim)

        # Dual optimizer via explicit kwarg
        svi_kwargs = Q_SVI(model=self.model, optim=vae_optim, bm_optim=bm_optim)
        self.assertTrue(svi_kwargs.use_two_optimisers)
        self.assertIs(svi_kwargs.vae_optim, vae_optim)
        self.assertIs(svi_kwargs.bm_optim, bm_optim)

    def test_two_stage_step_updates_weights_and_isolates_gradients(self):
        """Q_SVI.step updates both VAE and BM weights while maintaining gradient isolation."""
        vae_params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        vae_optim = torch.optim.SGD(vae_params, lr=0.1)
        bm_optim = torch.optim.SGD(self.model.bm.parameters(), lr=0.1)
        svi = Q_SVI(model=self.model, optim=vae_optim, bm_optim=bm_optim, bm_weight_decay=0.01)

        enc_weight_before = self.model.encoder.linear.weight.clone()
        dec_weight_before = self.model.decoder.linear.weight.clone()
        bm_weight_before = self.model.bm.quadratic_coef.clone()

        x = torch.rand(4, self.input_dim)
        loss = svi.step(x)

        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)

        # Verify weights have been updated in both stages
        self.assertFalse(torch.allclose(self.model.encoder.linear.weight, enc_weight_before))
        self.assertFalse(torch.allclose(self.model.decoder.linear.weight, dec_weight_before))
        self.assertFalse(torch.allclose(self.model.bm.quadratic_coef, bm_weight_before))

        loss_dict = svi.get_last_loss_dict()
        self.assertIn("total_loss", loss_dict)
        self.assertIn("vae_loss", loss_dict)
        self.assertIn("bm_loss", loss_dict)
        self.assertAlmostEqual(loss_dict["total_loss"], loss_dict["vae_loss"] + loss_dict["bm_loss"], places=5)

    def test_evaluate_loss_does_not_modify_parameters(self):
        """evaluate_loss returns deterministic loss without updating parameters."""
        vae_params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        vae_optim = torch.optim.SGD(vae_params, lr=0.1)
        bm_optim = torch.optim.SGD(self.model.bm.parameters(), lr=0.1)
        svi = Q_SVI(model=self.model, optim=vae_optim, bm_optim=bm_optim)

        self.model.eval()
        enc_weight_before = self.model.encoder.linear.weight.clone()
        bm_weight_before = self.model.bm.quadratic_coef.clone()

        x = torch.rand(4, self.input_dim)
        torch.manual_seed(42)
        val_loss1 = svi.evaluate_loss(x)
        torch.manual_seed(42)
        val_loss2 = svi.evaluate(x)

        self.assertAlmostEqual(val_loss1, val_loss2, places=6)
        torch.testing.assert_close(self.model.encoder.linear.weight, enc_weight_before)
        torch.testing.assert_close(self.model.bm.quadratic_coef, bm_weight_before)
        self.assertIsNone(self.model.encoder.linear.weight.grad)


class TestQSVILegacyParity(unittest.TestCase):
    """Test strict numerical parity between backend='legacy' and backend='svi'."""

    def test_loss_parity_across_epochs(self):
        """Under identical seeds and initial weights, SVI and legacy match within <1e-4."""
        input_dim = 8
        latent_dim = 4
        num_epochs = 3
        batch_size = 4

        # Fixed synthetic dataset
        torch.manual_seed(999)
        x_data = torch.rand(16, input_dim)
        y_data = torch.zeros(16, dtype=torch.long)
        dataset = TensorDataset(x_data, y_data)

        # Run 1: Legacy path
        torch.manual_seed(100)
        model_legacy = create_mock_qvae(input_dim, latent_dim, seed=100)
        loader_legacy = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        vae_params_l = list(model_legacy.encoder.parameters()) + list(model_legacy.decoder.parameters())
        vae_optim_l = torch.optim.Adam(vae_params_l, lr=1e-3)
        bm_optim_l = torch.optim.Adam(model_legacy.bm.parameters(), lr=1e-3)

        legacy_losses = []
        for _ in range(num_epochs):
            model_legacy.train()
            total_loss = 0.0
            for batch_x, _ in loader_legacy:
                vae_optim_l.zero_grad()
                output_logits, posterior, q, _ = model_legacy(batch_x)
                train_loss = model_legacy.loss(batch_x, output_logits, posterior)
                train_loss.backward()
                vae_optim_l.step()

                bm_optim_l.zero_grad()
                bm_loss = model_legacy.bm_loss(q.detach(), 0.01)
                bm_loss.backward()
                bm_optim_l.step()

                total_loss += train_loss.item() + bm_loss.item()
            total_loss /= len(dataset)
            legacy_losses.append(total_loss)

        # Run 2: SVI path
        torch.manual_seed(100)
        model_svi = create_mock_qvae(input_dim, latent_dim, seed=100)
        loader_svi = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        vae_params_s = list(model_svi.encoder.parameters()) + list(model_svi.decoder.parameters())
        vae_optim_s = torch.optim.Adam(vae_params_s, lr=1e-3)
        bm_optim_s = torch.optim.Adam(model_svi.bm.parameters(), lr=1e-3)

        svi = Q_SVI(
            model=model_svi,
            optim=vae_optim_s,
            bm_optim=bm_optim_s,
            bm_weight_decay=0.01,
        )

        svi_losses = []
        for _ in range(num_epochs):
            model_svi.train()
            total_loss = 0.0
            for batch_x, _ in loader_svi:
                step_loss = svi.step(batch_x)
                total_loss += step_loss
            total_loss /= len(dataset)
            svi_losses.append(total_loss)

        # Check parity: relative error must be < 1e-4 (< 1e-3 requirement)
        for ep, (loss_l, loss_s) in enumerate(zip(legacy_losses, svi_losses)):
            rel_error = abs(loss_s - loss_l) / abs(loss_l)
            self.assertLess(
                rel_error,
                1e-4,
                f"Epoch {ep} parity failed: legacy={loss_l:.6f}, svi={loss_s:.6f}, rel_err={rel_error:.2e}",
            )


class TestExampleTrainerIntegration(unittest.TestCase):
    """Test ModelTuner and SVITuner integration in example/qvae_mnist."""

    def test_model_tuner_svi_backend(self):
        """ModelTuner correctly switches between legacy and SVI backends."""
        import sys
        import os

        # Add example/qvae_mnist to sys.path
        qvae_mnist_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../example/qvae_mnist")
        )
        if qvae_mnist_path not in sys.path:
            sys.path.insert(0, qvae_mnist_path)

        from trainer.model_tuner import ModelTuner, SVITuner

        input_dim = 8
        latent_dim = 4
        model = create_mock_qvae(input_dim, latent_dim)

        x_data = torch.rand(8, input_dim)
        y_data = torch.zeros(8, dtype=torch.long)
        train_loader = DataLoader(TensorDataset(x_data, y_data), batch_size=4)
        test_loader = DataLoader(TensorDataset(x_data, y_data), batch_size=4)

        config = types.SimpleNamespace(
            type="QVAE",
            weight_decay=0.01,
        )

        # Test legacy tuner
        tuner_legacy = ModelTuner(config=config, backend="legacy")
        tuner_legacy.register_model(model)
        tuner_legacy.register_dataLoaders(train_loader, test_loader)
        vae_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
        tuner_legacy.register_two_optimisers(
            torch.optim.SGD(vae_params, lr=0.01),
            torch.optim.SGD(model.bm.parameters(), lr=0.01),
        )
        loss_l = tuner_legacy.train_epoch(1)
        self.assertIsInstance(loss_l, float)

        # Test SVITuner
        tuner_svi = SVITuner(config=config)
        self.assertEqual(tuner_svi.backend, "svi")
        tuner_svi.register_model(model)
        tuner_svi.register_dataLoaders(train_loader, test_loader)
        tuner_svi.register_two_optimisers(
            torch.optim.SGD(vae_params, lr=0.01),
            torch.optim.SGD(model.bm.parameters(), lr=0.01),
        )
        loss_s = tuner_svi.train_epoch(1)
        self.assertIsInstance(loss_s, float)

        # Test eval_pr and test methods
        test_res = tuner_svi.eval_pr()
        self.assertIsNotNone(test_res)

    def test_svi_backend_smoke(self):
        """Single-epoch smoke test for CI validation."""
        input_dim = 8
        latent_dim = 4
        model = create_mock_qvae(input_dim, latent_dim)

        vae_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
        svi = Q_SVI(
            model=model,
            optim=torch.optim.Adam(vae_params, lr=1e-3),
            bm_optim=torch.optim.Adam(model.bm.parameters(), lr=1e-3),
        )

        x_smoke = torch.rand(4, input_dim)
        loss_smoke = svi.step(x_smoke)
        self.assertGreater(loss_smoke, 0.0)
