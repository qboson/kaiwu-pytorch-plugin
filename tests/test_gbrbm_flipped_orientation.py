"""Tests for the flipped GBRBM orientation (is_visible_gaussian=False).

By default the GBRBM puts its Gaussian units on the visible side. With
``is_visible_gaussian=False`` the roles swap: the *hidden* side is Gaussian
and the *visible* side is Bernoulli. Every method indexes states as
``[gaussian | bernoulli]`` regardless of which side is visible, so the
flipped orientation exercises different index arithmetic than the default
one — and had zero test coverage (tests/test_gbrbm.py only builds default
models). These tests lock the flipped path in: parameter layout, energy
consistency against the closed-form expression, the infer/gibbs/sample
round trips, and the Bernoulli-side Ising equivalence.
"""

import itertools
import os
import sys
import unittest

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

import numpy as np
import torch

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


def make_flipped_model(seed=0):
    """Build a flipped-orientation GBRBM with fixed parameters."""
    torch.manual_seed(seed)
    model = GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=3,
        num_hidden=2,
        is_visible_gaussian=False,
        device=torch.device("cpu"),
    )
    return model


class TestFlippedLayout(unittest.TestCase):
    """Parameter layout of the flipped orientation."""

    def test_sides_and_shapes(self):
        """The hidden side is Gaussian and the visible side is Bernoulli."""
        model = make_flipped_model()
        self.assertEqual(model.num_gaussian, model.num_hidden)
        self.assertEqual(model.num_bernoulli, model.num_visible)
        self.assertEqual(model.num_nodes, 5)
        self.assertEqual(tuple(model.mu.shape), (model.num_hidden,))
        self.assertEqual(tuple(model.quadratic_coef.shape), (2, 3))
        self.assertEqual(tuple(model.linear_bias.shape), (3,))


class TestFlippedEnergy(unittest.TestCase):
    """Energy evaluation for full states in the flipped orientation."""

    def test_forward_matches_closed_form(self):
        """forward() equals the energy formula evaluated on [gaussian|bernoulli]."""
        model = make_flipped_model(seed=1)
        states = torch.zeros(8, 5)
        states[:, :2] = torch.randn(8, 2)
        states[:, 2:] = (torch.rand(8, 3) > 0.5).float()

        s_g = states[:, : model.num_gaussian]
        s_b = states[:, model.num_gaussian :]
        expected = (
            0.5 * torch.sum((s_g - model.mu).square() / model.var, dim=-1)
            - torch.sum((s_g / model.var) @ model.quadratic_coef * s_b, dim=-1)
            - s_b @ model.linear_bias
        )
        self.assertTrue(torch.allclose(model(states), expected, atol=1e-6))

    def test_energy_is_finite_for_random_states(self):
        """Random full states produce finite energies."""
        model = make_flipped_model(seed=2)
        states = torch.zeros(16, 5)
        states[:, :2] = torch.randn(16, 2)
        states[:, 2:] = (torch.rand(16, 3) > 0.5).float()
        self.assertTrue(torch.isfinite(model(states)).all())


class TestFlippedSampling(unittest.TestCase):
    """Inference and sampling paths in the flipped orientation."""

    def test_infer_from_gaussian(self):
        """Gaussian-side states complete to full states of the right width."""
        model = make_flipped_model(seed=3)
        out = model.infer_from_gaussian(torch.randn(6, 2))
        self.assertEqual(tuple(out.shape), (6, 5))
        self.assertTrue(torch.isfinite(out).all())

    def test_infer_from_bernoulli(self):
        """Bernoulli-side states complete to full states of the right width."""
        model = make_flipped_model(seed=4)
        out = model.infer_from_bernoulli(torch.randint(0, 2, (6, 3)).float())
        self.assertEqual(tuple(out.shape), (6, 5))
        self.assertTrue(torch.isfinite(out).all())

    def test_gibbs_sample_runs(self):
        """Gibbs sampling alternates both sides without shape errors."""
        model = make_flipped_model(seed=5)
        samples = model.gibbs_sample(n_step=5, n_burnin=2, n_sample=6)
        self.assertEqual(samples.ndim, 2)
        self.assertEqual(samples.shape[1], 5)
        self.assertTrue(torch.isfinite(samples).all())

    def test_sample_with_solver(self):
        """sample() solves the Bernoulli (visible-side) Ising and infers."""
        model = make_flipped_model(seed=6)

        class DummySampler:
            """Return fixed spin solutions with the auxiliary gauge spin."""

            def solve(self, ising_mat):
                """Return all-ones spins for any Ising matrix."""
                self.ising_shape = ising_mat.shape
                return np.ones((2, ising_mat.shape[0]), dtype=np.float32)

        sampler = DummySampler()
        out = model.sample(sampler)
        self.assertEqual(tuple(out.shape), (2, 5))
        # The Bernoulli side has num_visible=3 units -> (3+1)x(3+1) Ising.
        self.assertEqual(sampler.ising_shape, (4, 4))


class TestFlippedIsingEquivalence(unittest.TestCase):
    """Bernoulli-side Ising conversion in the flipped orientation."""

    def test_equivalence_over_all_states(self):
        """E_eff(h) + s^T M s is constant over all Bernoulli states.

        The effective energy comes from analytically integrating out the
        Gaussian units (completing the square), derived independently of
        _to_ising_matrix. The 1e-5 tolerance reflects the float32 default
        parameters of the model.
        """
        model = make_flipped_model(seed=7)
        num_bernoulli = model.num_bernoulli
        states = torch.tensor(
            list(itertools.product([0.0, 1.0], repeat=num_bernoulli))
        )
        precision = torch.diag(1.0 / model.var.detach())
        weights = model.quadratic_coef.detach()
        quadratic = weights.t() @ precision @ weights
        linear = weights.t() @ (model.mu.detach() / model.var.detach())
        linear = linear + model.linear_bias.detach()
        energies = (
            -0.5 * torch.einsum("bi,ij,bj->b", states, quadratic, states)
            - states @ linear
        )
        matrix = torch.tensor(model.get_ising_matrix())
        spins = 2.0 * states - 1.0
        spins_full = torch.cat([spins, torch.ones(len(spins), 1)], dim=1)
        ising_values = torch.einsum("bi,ij,bj->b", spins_full, matrix, spins_full)
        total = energies + ising_values
        spread = (total.max() - total.min()).item()
        self.assertLessEqual(spread, 1e-5, msg=f"spread was {spread:.3e}")


if __name__ == "__main__":
    unittest.main()
