"""Numerical-stability tests for the QVAE smoothing distributions."""

import sys
import unittest
import os

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin.qvae_dist_util import Exponential, MixtureGeneric


def reference_pdf(beta, zeta):
    """Float64 expm1 reference for the unit-interval exponential pdf."""
    b = torch.tensor(beta, dtype=torch.float64)
    z = zeta.to(torch.float64)
    return b * torch.exp(-b * z) / (-torch.expm1(-b))


def reference_cdf(beta, zeta):
    """Float64 expm1 reference for the unit-interval exponential cdf."""
    b = torch.tensor(beta, dtype=torch.float64)
    z = zeta.to(torch.float64)
    return (-torch.expm1(-b * z)) / (-torch.expm1(-b))


class TestExponentialSmallBeta(unittest.TestCase):
    """The smoothing distribution stays accurate as beta approaches zero."""

    def setUp(self):
        self.zeta = torch.linspace(0.05, 0.95, 7)

    def test_pdf_and_cdf_accurate_for_small_beta(self):
        """pdf/cdf match the float64 expm1 reference for small positive beta."""
        for beta in [1e-2, 1e-4, 1e-6, 1e-8]:
            dist = Exponential(beta)
            pdf = dist.pdf(self.zeta)
            cdf = dist.cdf(self.zeta)
            self.assertTrue(torch.isfinite(pdf).all(), f"pdf not finite at beta={beta}")
            self.assertTrue(torch.isfinite(cdf).all(), f"cdf not finite at beta={beta}")
            torch.testing.assert_close(
                pdf.to(torch.float64),
                reference_pdf(beta, self.zeta),
                rtol=1e-5,
                atol=1e-7,
            )
            torch.testing.assert_close(
                cdf.to(torch.float64),
                reference_cdf(beta, self.zeta),
                rtol=1e-5,
                atol=1e-7,
            )

    def test_log_pdf_accurate_for_small_beta(self):
        """log_pdf tends to the log-density of Uniform(0, 1) as beta -> 0."""
        for beta in [1e-6, 1e-8]:
            dist = Exponential(beta)
            log_pdf = dist.log_pdf(torch.tensor(0.5))
            # exact value is -beta/2 + O(beta^2), i.e. ~0
            self.assertTrue(torch.isfinite(log_pdf).all())
            self.assertLess(abs(log_pdf.item()), 1e-5)

    def test_sample_approaches_uniform_for_small_beta(self):
        """Samples stay ~Uniform(0, 1) for beta below the float32 epsilon."""
        torch.manual_seed(0)
        dist = Exponential(1e-8)
        samples = dist.sample((20000,))
        self.assertTrue(torch.isfinite(samples).all())
        self.assertGreater(samples.max().item(), 0.5)
        self.assertLess(abs(samples.mean().item() - 0.5), 0.01)
        self.assertLess(abs(samples.std().item() - 0.2887), 0.01)

    def test_mixture_reparameterize_gradients_finite_for_small_beta(self):
        """MixtureGeneric reparameterization stays finite for tiny beta."""
        torch.manual_seed(0)
        logits = torch.randn(4, 3, requires_grad=True)
        mixture = MixtureGeneric(logits, 1e-8)
        zeta = mixture.reparameterize(is_training=False)
        zeta.sum().backward()
        self.assertTrue(torch.isfinite(zeta).all())
        self.assertTrue(torch.isfinite(logits.grad).all())


class TestExponentialBackwardCompatibility(unittest.TestCase):
    """Behavior is unchanged for the beta values used in practice."""

    def test_pdf_cdf_match_direct_formula_for_moderate_beta(self):
        """For moderate beta the results equal the naive formula to rounding."""
        beta = 10.0  # default dist_beta in the QVAE examples
        dist = Exponential(beta)
        zeta = torch.linspace(0.05, 0.95, 7)
        b = torch.tensor(beta, dtype=torch.float32)

        naive_pdf = b * torch.exp(-b * zeta) / (1 - torch.exp(-b))
        naive_cdf = (1 - torch.exp(-b * zeta)) / (1 - torch.exp(-b))

        torch.testing.assert_close(dist.pdf(zeta), naive_pdf)
        torch.testing.assert_close(dist.cdf(zeta), naive_cdf)

    def test_sample_distribution_matches_cdf_for_moderate_beta(self):
        """Empirical CDF of sample() agrees with cdf() for beta=2."""
        torch.manual_seed(0)
        beta = 2.0
        dist = Exponential(beta)
        samples = dist.sample((20000,))
        grid = torch.linspace(0.0, 1.0, 101)
        empirical = torch.searchsorted(
            samples.sort().values, grid, right=True
        ).float() / samples.numel()
        analytic = dist.cdf(grid)
        max_gap = (empirical - analytic).abs().max().item()
        self.assertLess(max_gap, 0.02)


if __name__ == "__main__":
    unittest.main()
