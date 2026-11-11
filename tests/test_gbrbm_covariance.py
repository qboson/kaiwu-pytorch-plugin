import os
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

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


class TestGBRBMFullCovariance(unittest.TestCase):
    """Tests for the full-covariance Gaussian units of the GBRBM."""

    def setUp(self) -> None:
        """Build a deterministic full-covariance GBRBM."""
        self.covariance = torch.tensor(
            [[2.0, 0.6, 0.0], [0.6, 1.5, -0.4], [0.0, -0.4, 1.0]],
            dtype=torch.float64,
        )
        self.bm = GaussianBernoulliRestrictedBoltzmannMachine(
            num_visible=3,
            num_hidden=2,
            covariance=self.covariance,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        self.bm.mu.data = torch.tensor([1.0, -0.5, 0.25], dtype=torch.float64)
        self.bm.quadratic_coef.data = torch.tensor(
            [[2.0, -1.0], [0.5, 1.0], [-0.75, 0.3]], dtype=torch.float64
        )
        self.bm.linear_bias.data = torch.tensor([0.25, -0.5], dtype=torch.float64)

    def precision(self) -> torch.Tensor:
        """Explicit inverse covariance for cross-checking the fast paths."""
        return torch.linalg.inv(self.covariance)

    def test_covariance_roundtrip_buffers(self):
        """cov_chol and the trainable diagonal reconstruct the covariance."""
        var = self.bm.log_var.data.exp()
        reconstructed = self.bm.cov_chol @ torch.diag(var) @ self.bm.cov_chol.t()
        torch.testing.assert_close(reconstructed, self.covariance)
        legacy = GaussianBernoulliRestrictedBoltzmannMachine(
            num_visible=3, num_hidden=2, dtype=torch.float64
        )
        self.assertIsNone(legacy.cov_chol)
        self.assertIsNone(legacy.cov_chol_inv)

    def test_precision_scaled_matches_explicit_inverse(self):
        """_precision_scaled equals right-multiplication by inv(covariance)."""
        s = torch.tensor([[0.3, -1.2, 2.0], [1.5, 0.4, -0.7]], dtype=torch.float64)
        lam = self.precision()
        torch.testing.assert_close(self.bm._precision_scaled(s), s @ lam)
        torch.testing.assert_close(
            self.bm._precision_scaled(self.bm.mu.data), self.bm.mu.data @ lam
        )

    def test_gaussian_prior_energy_matches_quadratic_form(self):
        """The Gaussian prior term equals 0.5*(v-mu)^T inv(cov) (v-mu)."""
        s_gaussian = torch.tensor([[0.3, -1.2, 2.0], [1.5, 0.4, -0.7]], dtype=torch.float64)
        centered = s_gaussian - self.bm.mu.data
        expected = 0.5 * ((centered @ self.precision()) * centered).sum(dim=-1)
        torch.testing.assert_close(
            self.bm._gaussian_prior_energy(s_gaussian), expected
        )

    def test_energy_matches_full_covariance_formula(self):
        """energy reproduces the full-precision Hamiltonian."""
        s_gaussian = torch.tensor([[0.3, -1.2, 2.0], [1.5, 0.4, -0.7]], dtype=torch.float64)
        s_bernoulli = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        s_all = torch.cat([s_gaussian, s_bernoulli], dim=1)
        lam = self.precision()
        centered = s_gaussian - self.bm.mu.data
        expected = (
            0.5 * ((centered @ lam) * centered).sum(dim=-1)
            - (
                (s_gaussian @ lam) @ self.bm.quadratic_coef.data * s_bernoulli
            ).sum(dim=-1)
            - s_bernoulli @ self.bm.linear_bias.data
        )
        torch.testing.assert_close(self.bm.energy(s_all), expected)

    def test_marginal_energy_matches_formula(self):
        """marginal_energy uses the full-covariance Gaussian prior."""
        s_gaussian = torch.tensor([[0.3, -1.2, 2.0], [1.0, 0.1, -0.2]], dtype=torch.float64)
        lam = self.precision()
        centered = s_gaussian - self.bm.mu.data
        expected = (
            0.5 * ((centered @ lam) * centered).sum(dim=-1)
            - F.softplus(
                (s_gaussian @ lam) @ self.bm.quadratic_coef.data
                + self.bm.linear_bias.data
            ).sum(dim=-1)
        )
        torch.testing.assert_close(self.bm.marginal_energy(s_gaussian), expected)

    def test_positive_phase_energy_expectation_matches_formula(self):
        """positive_phase_energy_expectation uses the full precision."""
        s_gaussian = torch.tensor([[0.3, -1.2, 2.0], [1.0, 0.1, -0.2]], dtype=torch.float64)
        lam = self.precision()
        centered = s_gaussian - self.bm.mu.data
        prob = torch.sigmoid(
            (s_gaussian @ lam) @ self.bm.quadratic_coef.data
            + self.bm.linear_bias.data
        )
        expected = (
            0.5 * ((centered @ lam) * centered).sum(dim=-1)
            - ((s_gaussian @ lam) @ self.bm.quadratic_coef.data * prob).sum(dim=-1)
            - prob @ self.bm.linear_bias.data
        )
        torch.testing.assert_close(
            self.bm.positive_phase_energy_expectation(s_gaussian), expected
        )

    def test_infer_from_gaussian_matches_full_precision_logit(self):
        """Hidden probabilities use the full-precision logits."""
        s_gaussian = torch.tensor([[0.3, -1.2, 2.0], [1.0, 0.1, -0.2]], dtype=torch.float64)
        expected = torch.sigmoid(
            (s_gaussian @ self.precision()) @ self.bm.quadratic_coef.data
            + self.bm.linear_bias.data
        )
        prob = self.bm.infer_from_gaussian(s_gaussian, binarize=False)[
            :, self.bm.num_gaussian :
        ]
        torch.testing.assert_close(prob, expected)

    def test_infer_from_bernoulli_noise_covariance(self):
        """Random Gaussian samples reproduce the supplied covariance."""
        torch.manual_seed(0)
        s_bernoulli = torch.zeros(20000, 2, dtype=torch.float64)
        states = self.bm.infer_from_bernoulli(s_bernoulli)
        samples = states[:, : self.bm.num_gaussian].numpy()
        np.testing.assert_allclose(
            samples.mean(axis=0), self.bm.mu.data.numpy(), atol=0.05
        )
        np.testing.assert_allclose(np.cov(samples.T), self.covariance.numpy(), atol=0.1)

    def test_infer_from_bernoulli_no_random_uses_mean(self):
        """no_random=True returns the conditional mean without noise."""
        s_bernoulli = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        states = self.bm.infer_from_bernoulli(s_bernoulli, no_random=True)
        expected = (
            s_bernoulli @ self.bm.quadratic_coef.data.t() + self.bm.mu.data
        )
        torch.testing.assert_close(states[:, : self.bm.num_gaussian], expected)

    def test_to_ising_matrix_matches_formula(self):
        """The Bernoulli-side Ising matrix uses the full precision."""
        ising = self.bm._to_ising_matrix()
        lam = self.precision()
        w = self.bm.quadratic_coef.data
        b = self.bm.linear_bias.data
        linear_term = w.t() @ (lam @ self.bm.mu.data) + b
        quadratic_term = w.t() @ lam @ w
        column_sums = quadratic_term.sum(dim=0)
        n = self.bm.num_bernoulli
        expected = np.zeros((n + 1, n + 1))
        expected[:n, :n] = (quadratic_term * 0.125).numpy()
        expected[:n, n] = (linear_term * 0.25 + column_sums * 0.125).numpy()
        expected[n, :n] = expected[:n, n]
        np.testing.assert_allclose(ising, expected)

    def test_gibbs_sample_runs_with_full_covariance(self):
        """Gibbs sampling works end-to-end in the full-covariance mode."""
        torch.manual_seed(0)
        samples = self.bm.gibbs_sample(n_step=5, n_burnin=0, n_sample=16)
        self.assertEqual(samples.shape, (80, 5))
        self.assertTrue(torch.isfinite(samples).all().item())

    def test_gradients_flow_through_full_covariance(self):
        """The trainable parameters still receive finite gradients."""
        s_gaussian = torch.tensor([[0.3, -1.2, 2.0], [1.0, 0.1, -0.2]], dtype=torch.float64)
        loss = self.bm.positive_phase_energy_expectation(s_gaussian).mean()
        loss.backward()
        for param in (self.bm.log_var, self.bm.mu, self.bm.quadratic_coef, self.bm.linear_bias):
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all().item())
            self.assertGreater(param.grad.abs().sum().item(), 0.0)

    def test_diagonal_covariance_matches_legacy_model(self):
        """A diagonal covariance reduces exactly to the legacy behaviour."""
        d = np.array([4.0, 1.0, 2.5])
        torch.manual_seed(7)
        full = GaussianBernoulliRestrictedBoltzmannMachine(
            num_visible=3,
            num_hidden=2,
            covariance=np.diag(d),
            dtype=torch.float64,
        )
        torch.manual_seed(7)
        legacy = GaussianBernoulliRestrictedBoltzmannMachine(
            num_visible=3,
            num_hidden=2,
            dtype=torch.float64,
        )
        legacy.log_var.data = torch.log(torch.tensor(d, dtype=torch.float64))
        for attr in ("mu", "quadratic_coef", "linear_bias", "log_var"):
            torch.testing.assert_close(
                getattr(full, attr).data, getattr(legacy, attr).data
            )
        identity = torch.eye(3, dtype=torch.float64)
        torch.testing.assert_close(full.cov_chol, identity)
        torch.testing.assert_close(full.cov_chol_inv, identity)
        torch.manual_seed(3)
        s_all = torch.randn(6, 5, dtype=torch.float64)
        torch.testing.assert_close(full.energy(s_all), legacy.energy(s_all))
        np.testing.assert_allclose(full._to_ising_matrix(), legacy._to_ising_matrix())
        torch.manual_seed(11)
        samples_full = full.gibbs_sample(n_step=4, n_sample=8)
        torch.manual_seed(11)
        samples_legacy = legacy.gibbs_sample(n_step=4, n_sample=8)
        torch.testing.assert_close(samples_full, samples_legacy)

    def test_invalid_covariance_rejected(self):
        """Asymmetric, singular and wrongly shaped covariances raise ValueError."""
        with self.assertRaises(ValueError):
            GaussianBernoulliRestrictedBoltzmannMachine(
                num_visible=3,
                num_hidden=2,
                covariance=np.ones((3, 3)),
                dtype=torch.float64,
            )
        with self.assertRaises(ValueError):
            GaussianBernoulliRestrictedBoltzmannMachine(
                num_visible=3,
                num_hidden=2,
                covariance=np.array(
                    [[1.0, 0.5, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]]
                ),
                dtype=torch.float64,
            )
        with self.assertRaises(ValueError):
            GaussianBernoulliRestrictedBoltzmannMachine(
                num_visible=3,
                num_hidden=2,
                covariance=np.eye(2),
                dtype=torch.float64,
            )


if __name__ == "__main__":
    unittest.main()
