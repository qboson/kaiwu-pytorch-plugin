"""Regression tests for QVAE smoothing-distribution device placement.

``MixtureGeneric`` builds an ``Exponential`` smoothing distribution whose ``beta``
parameter is stored as a plain CPU tensor. During training reparameterization,
``pdf`` / ``cdf`` / ``log_pdf`` combine ``beta`` with ``zeta`` samples taken
from ``logit_mu`` on the accelerator. That mix raises a device mismatch on CUDA
machines. These tests pin the parameter-device query path used by
``qvae_dist_util.Exponential``.
"""

import unittest

import torch

from kaiwu.torch_plugin.qvae_dist_util import Exponential, MixtureGeneric


class TestExponentialDevicePlacement(unittest.TestCase):
    """Mechanism tests that run on every machine."""

    def test_pdf_matches_zeta_device_on_meta(self):
        """PDF outputs must live on the same device as the input ``zeta``."""
        smoothing = Exponential(1.0)
        zeta = torch.tensor([0.2, 0.7], device=torch.device("meta"))
        pdf = smoothing.pdf(zeta)
        self.assertEqual(pdf.device, zeta.device)

    def test_sample_honors_requested_device(self):
        """Ancestral sampling must allocate on the requested device."""
        smoothing = Exponential(0.5)
        zeta = smoothing.sample((3, 4), device=torch.device("meta"), dtype=torch.float32)
        self.assertEqual(zeta.device, torch.device("meta"))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestMixtureGenericCudaDevice(unittest.TestCase):
    """End-to-end CUDA coverage for mixture reparameterization."""

    def test_reparameterize_keeps_tensors_on_cuda(self):
        """Training reparameterization must not mix CPU smoothing params with CUDA logits."""
        logit_mu = torch.zeros(2, 5, device=torch.device("cuda"))
        posterior = MixtureGeneric(logit_mu, smoothing_dist_beta=1.0)
        zeta = posterior.reparameterize(is_training=True)
        self.assertEqual(zeta.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
