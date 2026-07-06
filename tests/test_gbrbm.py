import os
import sys
import unittest

import numpy as np
import torch

src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, src_root)

import kaiwu

# Extend namespace package __path__ so that local src is preferred
kaiwu.__path__ = list(kaiwu.__path__) + [os.path.join(src_root, "kaiwu")]

for module_name in list(sys.modules):
    if module_name == "kaiwu.torch_plugin" or module_name.startswith(
        "kaiwu.torch_plugin."
    ):
        del sys.modules[module_name]
if hasattr(kaiwu, "torch_plugin"):
    delattr(kaiwu, "torch_plugin")

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


class DummySampler:
    """Return fixed Ising spin solutions for GBRBM sampler tests."""

    def solve(self, ising_mat):
        """Return spin states with the last column as the gauge spin."""
        self.ising_shape = ising_mat.shape
        return np.array([[1, -1, 1], [-1, -1, -1]], dtype=np.float32)


class TestGaussianBernoulliRestrictedBoltzmannMachine(unittest.TestCase):
    """Unit tests for Gaussian-Bernoulli RBM."""

    def setUp(self) -> None:
        """Build a deterministic GBRBM for formula-based tests."""
        self.bm = GaussianBernoulliRestrictedBoltzmannMachine(
            num_visible=2,
            num_hidden=2,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        self.bm.mu.data = torch.tensor([1.0, -1.0], dtype=torch.float64)
        self.bm.log_var.data = torch.log(
            torch.tensor([4.0, 1.0], dtype=torch.float64)
        )
        self.bm.quadratic_coef.data = torch.tensor(
            [[2.0, -1.0], [0.5, 1.0]], dtype=torch.float64
        )
        self.bm.linear_bias.data = torch.tensor([0.25, -0.5], dtype=torch.float64)

    def test_constructor_matches_abstract_base(self):
        """GBRBM should initialize through the abstract base without dtype conflict."""
        self.assertEqual(self.bm.device, torch.device("cpu"))
        self.assertEqual(self.bm.dtype, torch.float64)
        self.assertEqual(self.bm.num_nodes, 4)
        self.assertEqual(self.bm.mu.dtype, torch.float64)

    def test_forward_energy_matches_manual_formula(self):
        """Forward should compute the Gaussian-Bernoulli energy."""
        s_all = torch.tensor(
            [[3.0, 0.0, 1.0, 0.0], [1.0, -2.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        s_gaussian = s_all[:, :2]
        s_bernoulli = s_all[:, 2:]
        expected = (
            0.5 * torch.sum((s_gaussian - self.bm.mu).square() / self.bm.var, dim=-1)
            - torch.sum(
                (s_gaussian / self.bm.var) @ self.bm.quadratic_coef * s_bernoulli,
                dim=-1,
            )
            - s_bernoulli @ self.bm.linear_bias
        )

        torch.testing.assert_close(self.bm(s_all), expected)

    def test_marginal_energy_matches_manual_formula(self):
        """Marginal energy should sum out the Bernoulli side analytically."""
        s_gaussian = torch.tensor([[3.0, 0.0], [1.0, -2.0]], dtype=torch.float64)
        logits = (
            (s_gaussian / self.bm.var) @ self.bm.quadratic_coef + self.bm.linear_bias
        )
        expected = (
            0.5 * torch.sum((s_gaussian - self.bm.mu).square() / self.bm.var, dim=-1)
            - torch.sum(torch.nn.functional.softplus(logits), dim=-1)
        )

        actual = self.bm.marginal_energy(s_gaussian)

        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(actual, expected)

    def test_positive_phase_energy_expectation_matches_manual_formula(self):
        """Positive phase should use Bernoulli probabilities from Gaussian data."""
        s_gaussian = torch.tensor([[3.0, 0.0], [1.0, -2.0]], dtype=torch.float64)
        prob_bernoulli = torch.sigmoid(
            (s_gaussian / self.bm.var) @ self.bm.quadratic_coef + self.bm.linear_bias
        )
        expected = (
            0.5 * torch.sum((s_gaussian - self.bm.mu).square() / self.bm.var, dim=-1)
            - torch.sum(
                (s_gaussian / self.bm.var) @ self.bm.quadratic_coef * prob_bernoulli,
                dim=-1,
            )
            - prob_bernoulli @ self.bm.linear_bias
        )

        torch.testing.assert_close(
            self.bm.positive_phase_energy_expectation(s_gaussian), expected
        )

    def test_positive_phase_can_disable_grad(self):
        """Positive phase should honor the enable_grad option."""
        s_gaussian = torch.tensor([[3.0, 0.0]], dtype=torch.float64)

        actual = self.bm.positive_phase_energy_expectation(
            s_gaussian,
            enable_grad=False,
        )

        self.assertFalse(actual.requires_grad)

    def test_get_ising_matrix_returns_bernoulli_side_matrix(self):
        """Ising matrix should cover only Bernoulli nodes plus gauge spin."""
        ising_mat = self.bm.get_ising_matrix()

        self.assertEqual(ising_mat.shape, (3, 3))
        np.testing.assert_allclose(ising_mat, ising_mat.T)

    def test_infer_from_gaussian_returns_probabilities(self):
        """Gaussian-side inference should expose Bernoulli probabilities."""
        s_gaussian = torch.tensor([[3.0, 0.0]], dtype=torch.float64)
        s_all = self.bm.infer_from_gaussian(s_gaussian, binarize=False)
        expected_prob = torch.sigmoid(
            (s_gaussian / self.bm.var) @ self.bm.quadratic_coef
            + self.bm.linear_bias
        )

        torch.testing.assert_close(s_all[:, :2], s_gaussian)
        torch.testing.assert_close(s_all[:, 2:], expected_prob)

    def test_infer_from_bernoulli_deterministic_mean(self):
        """Bernoulli-side deterministic inference should return Gaussian means."""
        s_bernoulli = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        s_all = self.bm.infer_from_bernoulli(s_bernoulli, no_random=True)
        expected_gaussian = s_bernoulli @ self.bm.quadratic_coef.t() + self.bm.mu

        torch.testing.assert_close(s_all[:, :2], expected_gaussian)
        torch.testing.assert_close(s_all[:, 2:], s_bernoulli)

    def test_sample_uses_abstract_sampler_contract(self):
        """sample(sampler) should rebuild full states from Bernoulli samples."""
        sampler = DummySampler()
        s_all = self.bm.sample(sampler)
        expected = torch.tensor(
            [[3.0, -0.5, 1.0, 0.0], [2.0, 0.5, 1.0, 1.0]],
            dtype=torch.float64,
        )

        self.assertEqual(sampler.ising_shape, (3, 3))
        torch.testing.assert_close(s_all, expected)

    def test_gibbs_sampling_supports_random_and_data_initialization(self):
        """The single Gibbs method should support generation and CD initialization."""
        torch.manual_seed(0)
        random_sample = self.bm.gibbs_sample(
            n_step=3,
            n_burnin=2,
            n_sample=2,
        )
        data_sample = self.bm.gibbs_sample(
            n_step=3,
            n_burnin=2,
            s_gaussian=torch.zeros(2, 2, dtype=torch.float64),
        )

        self.assertEqual(random_sample.shape, (4, 4))
        self.assertEqual(data_sample.shape, (4, 4))
        self.assertFalse(hasattr(self.bm, "conditional_gibbs_sample"))
        self.assertFalse(hasattr(self.bm, "cd_gibbs_sample"))

    def test_constructor_supports_bernoulli_visible_layout(self):
        """Non-Gaussian visible mode should still build Gaussian-first internals."""
        bm = GaussianBernoulliRestrictedBoltzmannMachine(
            num_visible=3,
            num_hidden=2,
            is_visible_gaussian=False,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )

        self.assertEqual(bm.num_gaussian, 2)
        self.assertEqual(bm.num_bernoulli, 3)
        self.assertEqual(bm.num_nodes, 5)
        self.assertEqual(bm.mu.shape, (2,))
        self.assertEqual(bm.quadratic_coef.shape, (2, 3))
        self.assertEqual(bm.linear_bias.shape, (3,))


if __name__ == "__main__":
    unittest.main()
