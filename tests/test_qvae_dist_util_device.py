# pylint: disable=protected-access
"""Regression tests for QVAE smoothing-distribution device placement.

``Exponential`` stores ``beta`` as a plain CPU tensor. ``MixtureGeneric`` keeps
``logit_mu`` on the accelerator and, during training reparameterization, feeds
``zeta`` samples into ``pdf`` / ``cdf`` / ``log_pdf``. Without co-locating
``beta``, PyTorch raises a cross-device error when ``zeta`` lives on CUDA (and
the same pattern is pinned on ``meta`` for CPU-only CI).

The CUDA-dependent tests below exercise the construct-on-CUDA training path that
CPU-only development machines cannot reproduce end-to-end.
"""

import unittest

import torch

from kaiwu.torch_plugin.qvae_dist_util import Exponential, MixtureGeneric


class TestExponentialDevicePlacement(unittest.TestCase):
    """Mechanism tests that run on every machine."""

    def test_beta_on_device_matches_reference(self):
        """Smoothing math must resolve ``beta`` from the active sample tensor."""
        beta = torch.tensor(2.0)
        reference = torch.zeros(3, device=torch.device("meta"))
        resolved = Exponential._beta_on_device(beta, reference)
        self.assertEqual(resolved.device, reference.device)
        self.assertEqual(resolved.dtype, reference.dtype)

    def test_pdf_cdf_and_log_pdf_match_zeta_device_on_meta(self):
        """Every smoothing evaluator must return tensors on ``zeta``'s device."""
        smoothing = Exponential(1.0)
        zeta = torch.tensor([0.2, 0.7], device=torch.device("meta"))
        self.assertEqual(smoothing.pdf(zeta).device, zeta.device)
        self.assertEqual(smoothing.cdf(zeta).device, zeta.device)
        self.assertEqual(smoothing.log_pdf(zeta).device, zeta.device)

    def test_sample_honors_requested_device(self):
        """Ancestral sampling must allocate on the requested device.

        On the pre-fix API ``sample(shape)`` rejected ``device`` / ``dtype`` and
        always drew on CPU, so this test fails before the fix (``TypeError``).
        """
        smoothing = Exponential(0.5)
        zeta = smoothing.sample(
            (3, 4),
            device=torch.device("meta"),
            dtype=torch.float32,
        )
        self.assertEqual(zeta.device, torch.device("meta"))

    def test_mixture_generic_reparameterize_on_meta(self):
        """Training reparameterization must keep the posterior sample on ``logit_mu``."""
        logit_mu = torch.zeros(2, 5, device=torch.device("meta"))
        posterior = MixtureGeneric(logit_mu, smoothing_dist_beta=1.0)
        zeta = posterior.reparameterize(is_training=True)
        self.assertEqual(zeta.device, logit_mu.device)
        self.assertEqual(zeta.shape, logit_mu.shape)

    def test_mixture_generic_log_prob_per_var_on_meta(self):
        """Mixture log-probability must not mix CPU smoothing params with meta logits."""
        logit_mu = torch.zeros(2, 4, device=torch.device("meta"))
        posterior = MixtureGeneric(logit_mu, smoothing_dist_beta=1.0)
        samples = torch.full_like(logit_mu, 0.25)
        log_prob = posterior.log_prob_per_var(samples)
        self.assertEqual(log_prob.device, logit_mu.device)
        self.assertEqual(log_prob.shape, logit_mu.shape)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDeviceSyncAcrossAccelerators(unittest.TestCase):
    """End-to-end coverage of the CUDA training path."""

    def test_cross_device_multiply_is_rejected_without_co_location(self):
        """Document the PyTorch failure mode the fix prevents."""
        beta = torch.tensor(1.0)
        zeta = torch.tensor([0.5], device=torch.device("cuda"))
        with self.assertRaises(RuntimeError):
            _ = beta * torch.exp(-beta * zeta)

    def test_reparameterize_keeps_tensors_on_cuda(self):
        """Training reparameterization must not mix CPU smoothing params with CUDA logits."""
        logit_mu = torch.zeros(2, 5, device=torch.device("cuda"))
        posterior = MixtureGeneric(logit_mu, smoothing_dist_beta=1.0)
        zeta = posterior.reparameterize(is_training=True)
        self.assertEqual(zeta.device.type, "cuda")

    def test_log_prob_per_var_on_cuda(self):
        """Mixture log-probability must run entirely on the logit device."""
        logit_mu = torch.randn(3, 6, device=torch.device("cuda"))
        posterior = MixtureGeneric(logit_mu, smoothing_dist_beta=0.75)
        samples = torch.rand_like(logit_mu)
        log_prob = posterior.log_prob_per_var(samples)
        self.assertEqual(log_prob.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
