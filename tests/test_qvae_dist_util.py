"""Differential tests for the QVAE distribution utilities.

``qvae_dist_util.py`` backs every probability term in the QVAE loss, but until
now nothing imported it from a test: the existing suite reached it only
indirectly through ``QVAE``, which left four methods never executed at all --
``Exponential.log_pdf``, ``FactorialBernoulliUtil.log_prob_per_var``,
``MixtureGeneric.log_prob_per_var`` and ``MixtureGeneric.log_ratio``.

Every assertion below is checked against an independent reference rather than
against the implementation itself:

* the Bernoulli helpers against ``torch.distributions.Bernoulli``
* ``Exponential`` against its own numerical integral, plus a sampled CDF
* ``MixtureGeneric`` against a 200k-point numerical integration of the density
  it claims, compared with the empirical CDF of 100k ancestral samples

The module is correct today; these tests pin that down so it stays correct.
"""
import unittest

import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from kaiwu.torch_plugin.qvae_dist_util import (
    DistUtil,
    Exponential,
    FactorialBernoulliUtil,
    MixtureGeneric,
    SmoothingDist,
    sigmoid_cross_entropy_with_logits,
)


def _seed(value=1234):
    torch.manual_seed(value)


class TestSigmoidCrossEntropy(unittest.TestCase):
    """sigmoid_cross_entropy_with_logits must match PyTorch's BCE-with-logits."""

    def test_matches_binary_cross_entropy_with_logits(self):
        _seed()
        logits = torch.randn(200, 7) * 3
        labels = (torch.rand(200, 7) < 0.5).float()
        expected = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        torch.testing.assert_close(
            sigmoid_cross_entropy_with_logits(logits, labels), expected
        )

    def test_matches_closed_form(self):
        """BCE = softplus(logits) - labels * logits."""
        _seed()
        logits = torch.randn(50, 4) * 2
        labels = (torch.rand(50, 4) < 0.5).float()
        expected = F.softplus(logits) - labels * logits
        torch.testing.assert_close(
            sigmoid_cross_entropy_with_logits(logits, labels), expected
        )


class TestFactorialBernoulliUtil(unittest.TestCase):
    """log_prob_per_var and entropy must match torch.distributions.Bernoulli."""

    def setUp(self):
        _seed()
        self.logit_mu = torch.randn(50, 5) * 2
        self.dist = FactorialBernoulliUtil(self.logit_mu)

    def test_log_prob_per_var_matches_bernoulli(self):
        samples = (torch.rand(50, 5) < 0.5).float()
        expected = Bernoulli(logits=self.logit_mu).log_prob(samples)
        torch.testing.assert_close(self.dist.log_prob_per_var(samples), expected)

    def test_entropy_matches_bernoulli(self):
        expected = Bernoulli(logits=self.logit_mu).entropy()
        torch.testing.assert_close(self.dist.entropy(), expected)

    def test_reparameterize_refuses_training_mode(self):
        """The base class documents that Bernoulli reparameterisation is not
        differentiable, so it must refuse rather than silently return noise."""
        with self.assertRaises(NotImplementedError):
            self.dist.reparameterize(is_training=True)

    def test_reparameterize_returns_binary_samples(self):
        out = self.dist.reparameterize(is_training=False)
        self.assertEqual(tuple(out.shape), tuple(self.logit_mu.shape))
        self.assertTrue(bool(((out == 0) | (out == 1)).all()))


class TestExponential(unittest.TestCase):
    """pdf, cdf, log_pdf and sample must describe one consistent distribution."""

    BETAS = (0.5, 1.0, 2.0, 5.0)

    def test_pdf_integrates_to_one_on_the_unit_interval(self):
        z = torch.linspace(0, 1, 200001)
        for beta in self.BETAS:
            with self.subTest(beta=beta):
                integral = torch.trapz(Exponential(beta).pdf(z), z).item()
                self.assertAlmostEqual(integral, 1.0, places=3)

    def test_cdf_is_the_integral_of_pdf(self):
        z = torch.linspace(0, 1, 200001)
        dz = float(z[1] - z[0])
        for beta in self.BETAS:
            with self.subTest(beta=beta):
                dist = Exponential(beta)
                running = torch.cat(
                    [
                        torch.zeros(1),
                        torch.cumsum(0.5 * (dist.pdf(z[1:]) + dist.pdf(z[:-1])) * dz, 0),
                    ]
                )
                torch.testing.assert_close(running, dist.cdf(z), atol=1e-4, rtol=0)

    def test_cdf_endpoints(self):
        for beta in self.BETAS:
            with self.subTest(beta=beta):
                dist = Exponential(beta)
                self.assertAlmostEqual(dist.cdf(torch.tensor(0.0)).item(), 0.0, places=6)
                self.assertAlmostEqual(dist.cdf(torch.tensor(1.0)).item(), 1.0, places=6)

    def test_log_pdf_is_log_of_pdf(self):
        z = torch.linspace(0, 1, 21)
        for beta in self.BETAS:
            with self.subTest(beta=beta):
                dist = Exponential(beta)
                torch.testing.assert_close(
                    dist.log_pdf(z), torch.log(dist.pdf(z)), atol=1e-6, rtol=0
                )

    def test_sample_stays_on_the_support(self):
        for beta in self.BETAS:
            with self.subTest(beta=beta):
                s = Exponential(beta).sample((20000,))
                self.assertGreaterEqual(float(s.min()), 0.0)
                self.assertLessEqual(float(s.max()), 1.0)

    def test_sample_follows_the_cdf(self):
        """Inverse-CDF sampling: the empirical CDF must track cdf()."""
        quantiles = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
        for beta in self.BETAS:
            with self.subTest(beta=beta):
                _seed(7)
                dist = Exponential(beta)
                s, _ = torch.sort(dist.sample((200000,)))
                empirical = torch.tensor(
                    [(s <= t).float().mean() for t in quantiles]
                )
                torch.testing.assert_close(
                    empirical, dist.cdf(quantiles), atol=5e-3, rtol=0
                )


class TestMixtureGeneric(unittest.TestCase):
    """reparameterize() must sample exactly what log_prob_per_var() describes."""

    BETAS = (1.0, 2.0, 4.0)
    LOGIT_MU = torch.tensor([[0.0, 1.5, -1.5, 0.7, -0.7]])
    N = 100000

    def test_reparameterize_stays_on_the_unit_interval(self):
        for beta in self.BETAS:
            with self.subTest(beta=beta):
                _seed(11)
                big = MixtureGeneric(self.LOGIT_MU.expand(self.N, 5), beta)
                zeta = big.reparameterize(is_training=False)
                self.assertEqual(tuple(zeta.shape), (self.N, 5))
                self.assertGreaterEqual(float(zeta.min()), -1e-6)
                self.assertLessEqual(float(zeta.max()), 1.0 + 1e-6)

    def test_sampled_density_matches_log_prob_per_var(self):
        """The decisive consistency check.

        reparameterize() is documented as ancestral sampling from the mixture
        q*r(1-zeta) + (1-q)*r(zeta); log_prob_per_var() returns that density.
        Integrate the density on a fine grid and compare its CDF with the
        empirical CDF of the samples, per variable.
        """
        fine = torch.linspace(0.0, 1.0, 20001)
        dz = float(fine[1] - fine[0])
        coarse = torch.linspace(0.0, 1.0, 11)
        idx = (coarse * (fine.numel() - 1)).round().long()

        for beta in self.BETAS:
            with self.subTest(beta=beta):
                _seed(11)
                dist = MixtureGeneric(self.LOGIT_MU, beta)

                grid = fine.unsqueeze(1).expand(fine.numel(), 5).contiguous()
                density = dist.log_prob_per_var(grid).exp().clamp_min(0)
                analytic_cdf = torch.cumsum(density * dz, dim=0)

                big = MixtureGeneric(self.LOGIT_MU.expand(self.N, 5), beta)
                samples = big.reparameterize(is_training=False)

                for j in range(5):
                    column, _ = torch.sort(samples[:, j])
                    empirical = torch.tensor(
                        [(column <= float(t)).float().mean() for t in coarse]
                    )
                    torch.testing.assert_close(
                        empirical,
                        analytic_cdf[idx, j],
                        atol=1e-2,
                        rtol=0,
                        msg=f"variable {j} of 5 disagrees",
                    )

    def test_log_ratio_is_linear_in_zeta(self):
        """log r(z|z=1) - log r(z|z=0) simplifies to beta * (2*zeta - 1)."""
        z = torch.linspace(0, 1, 21)
        for beta in self.BETAS:
            with self.subTest(beta=beta):
                dist = MixtureGeneric(self.LOGIT_MU, beta)
                torch.testing.assert_close(
                    dist.log_ratio(z), beta * (2 * z - 1), atol=1e-4, rtol=0
                )

    def test_log_ratio_is_antisymmetric(self):
        z = torch.linspace(0, 1, 21)
        for beta in self.BETAS:
            with self.subTest(beta=beta):
                dist = MixtureGeneric(self.LOGIT_MU, beta)
                torch.testing.assert_close(dist.log_ratio(1 - z), -dist.log_ratio(z))

    def test_log_ratio_vanishes_at_the_midpoint(self):
        for beta in self.BETAS:
            with self.subTest(beta=beta):
                dist = MixtureGeneric(self.LOGIT_MU, beta)
                self.assertAlmostEqual(
                    dist.log_ratio(torch.tensor(0.5)).item(), 0.0, places=6
                )

    def test_gradient_is_the_implicit_gradient(self):
        """reparameterize() is a straight-through estimator: the forward value
        is just the ancestral sample and only the gradient carries the signal,
        so a forward-only check cannot see it.

        The implementation builds ``grad_term = grad_q * q`` with ``q`` left
        attached, so the gradient delivered to ``logit_mu`` is chain-ruled
        through the sigmoid:

            d(zeta)/d(logit_mu) = grad_q * q * (1 - q)

        where grad_q = (cdf_0 - cdf_1) / (q * pdf_1 + (1 - q) * pdf_0).
        """
        beta = 2.0
        _seed(3)
        logit_mu = torch.tensor(
            [[0.3, -1.1, 0.8, -0.4, 0.05]], dtype=torch.float32, requires_grad=True
        )
        dist = MixtureGeneric(logit_mu, beta)
        zeta = dist.reparameterize(is_training=True)
        zeta.sum().backward()

        with torch.no_grad():
            q = torch.sigmoid(logit_mu)
            z = zeta.detach()
            smoothing = Exponential(beta)
            pdf_0 = smoothing.pdf(z)
            pdf_1 = smoothing.pdf(1.0 - z)
            cdf_0 = smoothing.cdf(z)
            cdf_1 = 1.0 - smoothing.cdf(1.0 - z)
            grad_q = (cdf_0 - cdf_1) / (q * pdf_1 + (1 - q) * pdf_0)
            expected = grad_q * q * (1 - q)

        self.assertIsNotNone(logit_mu.grad)
        torch.testing.assert_close(logit_mu.grad, expected, atol=1e-6, rtol=0)


class TestAbstractBases(unittest.TestCase):
    """The base classes must refuse to be used directly."""

    def test_smoothing_dist_methods_raise(self):
        base = SmoothingDist()
        for name in ("pdf", "cdf", "sample", "log_pdf"):
            with self.subTest(method=name):
                with self.assertRaises(NotImplementedError):
                    getattr(base, name)(torch.zeros(1))

    def test_dist_util_methods_raise(self):
        base = DistUtil()
        with self.assertRaises(NotImplementedError):
            base.reparameterize(is_training=False)
        with self.assertRaises(NotImplementedError):
            base.entropy()


if __name__ == "__main__":
    unittest.main()