"""Q_SVI training-kernel tests: parity with the legacy two-phase loop.

The BM negative phase needs a sampler; the licensed kaiwu SDK samplers are
replaced by a seeded local stand-in with the same ``solve(ising_matrix)``
contract, so these regressions run offline.
"""

import sys
import os
import types
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin import Q_SVI, RestrictedBoltzmannMachine
from kaiwu.torch_plugin.qvae import QVAE


class SeededSampler:
    """Deterministic stand-in for kaiwu's SimulatedAnnealingOptimizer."""

    def __init__(self, seed=0, num_samples=4):
        self.rng = np.random.default_rng(seed)
        self.num_samples = num_samples

    def solve(self, ising_matrix):
        n = ising_matrix.shape[0]
        spins = self.rng.choice(
            [-1.0, 1.0], size=(self.num_samples, n - 1)
        ).astype(np.float32)
        # Kaiwu solvers append a value column; torch_plugin drops it.
        return np.hstack(
            [spins, np.zeros((self.num_samples, 1), dtype=np.float32)]
        )


class SmallQVAE(QVAE):
    """QVAE with tiny trainable encoder/decoder for offline regressions."""

    def _create_encoder(self):
        return nn.Sequential(
            nn.Linear(self._input_dimension, 16),
            nn.ReLU(),
            nn.Linear(16, self._latent_dimensions),
        )

    def _create_decoder(self):
        return nn.Sequential(
            nn.Linear(self._latent_dimensions, 16),
            nn.ReLU(),
            nn.Linear(16, self._input_dimension),
        )

    def _create_bm(self):
        num_visible = self._latent_dimensions // 2
        return RestrictedBoltzmannMachine(
            num_visible, self._latent_dimensions - num_visible
        )

    def _create_sampler(self, sampler_type):
        del sampler_type
        return self.sampler


class TestQSVI(unittest.TestCase):
    """Q_SVI parity, single-epoch smoke, and pyro-style interface coverage."""

    def setUp(self):
        self.input_dim = 16
        self.latent_dim = 8
        self.weight_decay = 0.01
        self.config = types.SimpleNamespace(
            num_latent_units=self.latent_dim,
            loss_type="bernoulli",
            dist_beta=2.0,
            kl_beta=1e-3,
            weight_decay=self.weight_decay,
            sampler_type="sa",
        )

    def _build(self, seed=7):
        torch.manual_seed(seed)
        model = SmallQVAE(
            input_dimension=self.input_dim,
            activation_fct=nn.ReLU(),
            config=self.config,
            sampler=SeededSampler(seed=seed + 1),
        )
        mean = torch.full((self.input_dim,), 0.25)
        model.set_dataset_mean(mean)
        model.set_train_bias(mean)
        vae_optim = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.decoder.parameters()),
            lr=1e-3,
        )
        bm_optim = torch.optim.Adam(model.bm.parameters(), lr=1e-3)
        return model, vae_optim, bm_optim

    def _loader(self, seed=11, num_samples=64, batch_size=16):
        generator = torch.Generator().manual_seed(seed)
        data = (torch.rand(num_samples, self.input_dim, generator=generator) > 0.5).float()
        dataset = TensorDataset(data, torch.zeros(num_samples, dtype=torch.long))
        # A fresh, identically seeded loader per epoch keeps shuffle order fixed.
        return lambda: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

    def _legacy_step(self, model, vae_optim, bm_optim, x):
        """Verbatim phase order of the legacy ModelTuner batch loop."""
        vae_optim.zero_grad()
        output_logits, posterior, q, _ = model(x)
        train_loss = model.loss(x, output_logits, posterior)
        train_loss.backward()
        vae_optim.step()
        batch_total = train_loss.item()
        if bm_optim is not None:
            bm_optim.zero_grad()
            bm_loss = model.bm_loss(q.detach(), self.weight_decay)
            bm_loss.backward()
            bm_optim.step()
            batch_total += bm_loss.item()
        return batch_total

    def test_step_matches_legacy_two_phase_loop(self):
        """Under fixed seeds Q_SVI reproduces the legacy loop batch-for-batch."""
        results = {}
        for backend in ("legacy", "svi"):
            torch.manual_seed(123)
            model, vae_optim, bm_optim = self._build()
            q_svi = Q_SVI(
                model=model,
                optim=vae_optim,
                bm_optim=bm_optim,
                bm_weight_decay=self.weight_decay,
            )
            make_loader = self._loader()
            batch_losses = []
            for _ in range(3):
                model.train()
                for x, _ in make_loader():
                    if backend == "legacy":
                        batch_losses.append(
                            self._legacy_step(model, vae_optim, bm_optim, x)
                        )
                    else:
                        batch_losses.append(q_svi.step(x))
            results[backend] = (batch_losses, model.state_dict())

        self.assertEqual(results["legacy"][0], results["svi"][0])
        for (key, legacy_value), (_, svi_value) in zip(
            results["legacy"][1].items(), results["svi"][1].items()
        ):
            self.assertTrue(
                torch.equal(legacy_value, svi_value), f"parameter mismatch: {key}"
            )

    def test_single_epoch_svi_backend_smoke(self):
        """One epoch through Q_SVI trains, returns finite floats, updates params."""
        model, vae_optim, bm_optim = self._build()
        initial = {
            key: value.clone() for key, value in model.state_dict().items()
        }
        q_svi = Q_SVI(
            model=model,
            optim=vae_optim,
            bm_optim=bm_optim,
            bm_weight_decay=self.weight_decay,
        )
        make_loader = self._loader()
        model.train()
        batch_losses = [q_svi.step(x) for x, _ in make_loader()]

        self.assertTrue(all(np.isfinite(batch_losses)))
        self.assertTrue(all(isinstance(value, float) for value in batch_losses))
        self.assertTrue(np.isfinite(q_svi.last_loss))
        self.assertTrue(np.isfinite(q_svi.last_bm_loss))
        changed = any(
            not torch.equal(initial[key], value)
            for key, value in model.state_dict().items()
        )
        self.assertTrue(changed, "one epoch of Q_SVI must update parameters")

    def test_single_stage_mode_matches_legacy(self):
        """Without bm_optim, step() runs the single-stage loop only."""
        results = {}
        for backend in ("legacy", "svi"):
            torch.manual_seed(321)
            model, vae_optim, _ = self._build()
            q_svi = Q_SVI(model=model, optim=vae_optim)
            make_loader = self._loader(seed=5)
            batch_losses = []
            for _ in range(2):
                model.train()
                for x, _ in make_loader():
                    if backend == "legacy":
                        vae_optim.zero_grad()
                        output_logits, posterior, _, _ = model(x)
                        loss = model.loss(x, output_logits, posterior)
                        loss.backward()
                        vae_optim.step()
                        batch_losses.append(loss.item())
                    else:
                        value = q_svi.step(x)
                        self.assertIsNone(q_svi.last_bm_loss)
                        batch_losses.append(value)
            results[backend] = batch_losses
        self.assertEqual(results["legacy"], results["svi"])

    def test_pyro_style_guide_and_loss_hooks(self):
        """Custom guide/loss callables are honored, kwargs reach the guide."""
        model, vae_optim, bm_optim = self._build()
        seen = {}

        def guide(x, **kwargs):
            seen["guide_kwargs"] = kwargs
            # Only forward kwargs the composed QVAE.forward understands.
            return model(x)

        def loss(x, recon_x, posterior):
            seen["loss_called"] = True
            return model.loss(x, recon_x, posterior)

        q_svi = Q_SVI(
            model=model,
            guide=guide,
            optim=vae_optim,
            loss=loss,
            bm_optim=bm_optim,
            bm_weight_decay=self.weight_decay,
        )
        x = torch.rand(4, self.input_dim)
        value = q_svi.step(x)
        self.assertEqual(seen["guide_kwargs"], {})
        self.assertTrue(seen["loss_called"])
        self.assertTrue(np.isfinite(value))

        q_svi.step(x, extra=1)
        self.assertEqual(seen["guide_kwargs"], {"extra": 1})

    def test_constructor_requires_optim(self):
        """A missing optim is rejected at construction time."""
        model, _, _ = self._build()
        with self.assertRaises(ValueError):
            Q_SVI(model=model)

    def test_loss_records_component_tensors(self):
        """QVAE.loss exposes detached recon/kl tensors for metric aggregation."""
        model, _, _ = self._build()
        output_logits, posterior, _, _ = model(torch.rand(4, self.input_dim))
        model.loss(torch.rand(4, self.input_dim), output_logits, posterior)
        self.assertTrue(torch.is_tensor(model.last_recon_loss))
        self.assertTrue(torch.is_tensor(model.last_kl_loss))
        model.last_recon_loss.item()
        model.last_kl_loss.item()


if __name__ == "__main__":
    unittest.main()
