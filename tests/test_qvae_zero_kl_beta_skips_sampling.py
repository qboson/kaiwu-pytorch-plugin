"""Regression tests for skipping the KL surrogate when ``kl_beta`` is zero."""

import types
import unittest

import numpy as np
import torch

from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.torch_plugin.qvae import QVAE
from kaiwu.torch_plugin.qvae_dist_util import FactorialBernoulliUtil


INPUT_DIMENSION = 8
LATENT_DIMENSION = 4
NUM_VISIBLE = LATENT_DIMENSION // 2
NUM_HIDDEN = LATENT_DIMENSION - NUM_VISIBLE
BATCH_SIZE = 5


class RecordingSampler:
    """Count how often the Boltzmann machine asks the solver for samples."""

    def __init__(self):
        self.solve_calls = 0

    def solve(self, ising_matrix):
        """Return a deterministic batch of spins and record the call."""
        self.solve_calls += 1
        return np.zeros((2, ising_matrix.shape[0]), dtype=np.float32)


class DummyEncoder(torch.nn.Module):
    """Trainable encoder, so reconstruction gradients reach real parameters."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(INPUT_DIMENSION, LATENT_DIMENSION)

    def forward(self, x):
        """Return the latent logits for a batch of inputs."""
        return self.linear(x)


class InjectedQVAE(QVAE):
    """QVAE whose components are injected, so the factories never run."""

    def _create_encoder(self):
        return self.encoder

    def _create_decoder(self):
        return self.decoder

    def _create_bm(self):
        return self.bm

    def _create_sampler(self, sampler_type):
        del sampler_type
        return self.sampler


def build_qvae(kl_beta, loss_type="bernoulli", weight_decay=0.0, seed=0):
    """Build a QVAE with an injected Boltzmann machine and a recording sampler."""
    torch.manual_seed(seed)
    config = types.SimpleNamespace(
        num_latent_units=LATENT_DIMENSION,
        loss_type=loss_type,
        dist_beta=1.0,
        kl_beta=kl_beta,
        weight_decay=weight_decay,
    )
    sampler = RecordingSampler()
    model = InjectedQVAE(
        input_dimension=INPUT_DIMENSION,
        activation_fct=None,
        config=config,
        encoder=DummyEncoder(),
        decoder=torch.nn.Linear(LATENT_DIMENSION, INPUT_DIMENSION),
        bm=RestrictedBoltzmannMachine(NUM_VISIBLE, NUM_HIDDEN),
        sampler=sampler,
    )
    return model, sampler


class TestZeroKlBetaSkipsSampling(unittest.TestCase):
    """The unweighted KL surrogate must not sample the Boltzmann machine."""

    def setUp(self):
        self.binary_batch = torch.randint(
            0, 2, (BATCH_SIZE, INPUT_DIMENSION)
        ).float()

    @staticmethod
    def _forward(model, batch):
        recon_x, posterior, logits, _ = model(batch)
        return recon_x, posterior, logits

    @staticmethod
    def _expected_weight_decay(model, weight_decay):
        quadratic = torch.sum(model.bm.quadratic_coef**2)
        linear = torch.sum(model.bm.linear_bias**2)
        return weight_decay * quadratic + weight_decay * 0.5 * linear

    def test_zero_kl_beta_skips_the_sampling(self):
        """A zero KL weight must not reach the Boltzmann machine sampler."""
        model, sampler = build_qvae(kl_beta=0.0)
        recon_x, posterior, _ = self._forward(model, self.binary_batch)
        sampler.solve_calls = 0

        total_loss = model.loss(self.binary_batch, recon_x, posterior)

        self.assertEqual(sampler.solve_calls, 0)
        self.assertTrue(torch.isfinite(total_loss))

        # The skipped term must stay per-sample, otherwise downstream reductions
        # would silently change shape.
        divergence = model._kl_dist_from(posterior)
        self.assertEqual(tuple(divergence.shape), (BATCH_SIZE,))
        self.assertTrue(torch.equal(divergence, torch.zeros(BATCH_SIZE)))

    def test_non_zero_kl_beta_still_samples_once(self):
        """A weighted KL term keeps sampling exactly once per loss call."""
        model, sampler = build_qvae(kl_beta=1e-3)
        recon_x, posterior, _ = self._forward(model, self.binary_batch)
        sampler.solve_calls = 0

        model.loss(self.binary_batch, recon_x, posterior)

        self.assertEqual(sampler.solve_calls, 1)

    def test_diagnostic_switch_restores_the_sampling(self):
        """``skip_zero_kl_beta = False`` evaluates the legacy surrogate on demand."""
        model, sampler = build_qvae(kl_beta=0.0)
        self.assertTrue(model.skip_zero_kl_beta)
        recon_x, posterior, _ = self._forward(model, self.binary_batch)
        sampler.solve_calls = 0

        model.skip_zero_kl_beta = False
        model.loss(self.binary_batch, recon_x, posterior)
        self.assertEqual(sampler.solve_calls, 1)

        model.skip_zero_kl_beta = True
        model.loss(self.binary_batch, recon_x, posterior)
        self.assertEqual(sampler.solve_calls, 1)

    def test_skipped_term_matches_a_vanishing_weight(self):
        """Skipping is safe: the loss equals the loss at ``kl_beta = 1e-12``."""
        model, sampler = build_qvae(kl_beta=0.0)
        recon_x, posterior, _ = self._forward(model, self.binary_batch)

        skipped = model.loss(self.binary_batch, recon_x, posterior)
        model.kl_beta = 1e-12
        vanishing = model.loss(self.binary_batch, recon_x, posterior)

        self.assertEqual(sampler.solve_calls, 1)
        torch.testing.assert_close(skipped, vanishing, rtol=1e-6, atol=1e-9)

        # The agreement must not be an artefact of a surrogate that is always zero.
        model.kl_beta = 1.0
        weighted = model.loss(self.binary_batch, recon_x, posterior)
        self.assertGreater(abs(float(weighted) - float(skipped)), 1e-3)

    def test_bm_loss_keeps_sampling_when_kl_beta_is_zero(self):
        """The independent Boltzmann machine update still needs its samples."""
        model, sampler = build_qvae(kl_beta=0.0)
        logits = torch.ones(BATCH_SIZE, LATENT_DIMENSION)
        sampler.solve_calls = 0

        bm_loss = model.bm_loss(logits)

        self.assertEqual(sampler.solve_calls, 1)
        self.assertTrue(torch.isfinite(bm_loss))

    def test_mse_reconstruction_gradient_and_weight_decay_survive(self):
        """Only the KL term is skipped; reconstruction and weight decay stay."""
        model, sampler = build_qvae(kl_beta=0.0, loss_type="mse", weight_decay=0.5)
        batch = torch.rand(BATCH_SIZE, INPUT_DIMENSION)
        recon_x, posterior, _ = self._forward(model, batch)
        sampler.solve_calls = 0

        total_loss = model.loss(batch, recon_x, posterior)
        expected_reconstruction = torch.nn.functional.mse_loss(
            recon_x, batch.view(-1, INPUT_DIMENSION), reduction="sum"
        ) / batch.size(0)
        expected_weight_decay = self._expected_weight_decay(model, 0.5)

        self.assertEqual(sampler.solve_calls, 0)
        self.assertGreater(float(expected_weight_decay), 0.0)
        torch.testing.assert_close(
            total_loss,
            expected_reconstruction + expected_weight_decay,
            rtol=1e-6,
            atol=1e-8,
        )

        total_loss.backward()
        gradients = [parameter.grad for parameter in model.encoder.parameters()]
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(
            any(float(gradient.abs().sum()) > 0.0 for gradient in gradients)
        )

    def test_bernoulli_reconstruction_term_survives(self):
        """The Bernoulli reconstruction term is unchanged by the skip."""
        model, sampler = build_qvae(kl_beta=0.0, loss_type="bernoulli")
        recon_x, posterior, _ = self._forward(model, self.binary_batch)
        sampler.solve_calls = 0

        total_loss = model.loss(self.binary_batch, recon_x, posterior)
        reconstruction = FactorialBernoulliUtil(recon_x)
        expected = -reconstruction.log_prob_per_var(self.binary_batch).sum(
            dim=1
        ).mean()

        self.assertEqual(sampler.solve_calls, 0)
        torch.testing.assert_close(total_loss, expected, rtol=1e-5, atol=1e-7)

    def test_positive_weight_matches_the_independent_bernoulli_entropy(self):
        """With a weight, the term is the sampled cross-entropy minus the entropy."""
        model, sampler = build_qvae(kl_beta=0.0)
        recon_x, posterior, logits = self._forward(model, self.binary_batch)
        skipped = model.loss(self.binary_batch, recon_x, posterior)
        model.kl_beta = 1.0
        weighted = model.loss(self.binary_batch, recon_x, posterior)

        probability = torch.sigmoid(logits)
        entropy = -(
            probability * torch.log(probability)
            + (1.0 - probability) * torch.log(1.0 - probability)
        ).sum(dim=1).mean()
        positive = model.bm(probability).mean()
        negative = model.bm(model.bm.sample(model.sampler)).mean()
        expected_surrogate = positive - negative - entropy

        self.assertGreater(sampler.solve_calls, 0)
        torch.testing.assert_close(
            weighted - skipped, expected_surrogate, rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
