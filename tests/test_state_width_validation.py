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

import torch

from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine


class TestRBMStateWidth(unittest.TestCase):
    """State-width validation on the RBM methods."""

    def setUp(self):
        self.rbm = RestrictedBoltzmannMachine(3, 2)

    def test_valid_widths_still_work(self):
        """Correct widths on every method keep working."""
        s_all = self.rbm.get_hidden(torch.rand(5, 3), bernoulli=True)
        self.assertEqual(tuple(s_all.shape), (5, 5))
        self.assertTrue(torch.isfinite(self.rbm(s_all)).all())
        back = self.rbm.get_visible(s_all[:, 3:])
        self.assertEqual(tuple(back.shape), (5, 5))

    def test_forward_wrong_width_fails_fast(self):
        """forward rejects wrong-width states with the expected shape."""
        for bad in (torch.rand(5, 4), torch.rand(5, 7)):
            with self.subTest(width=bad.shape[1]):
                with self.assertRaises(ValueError) as ctx:
                    self.rbm(bad)
                self.assertIn("(batch, 5)", str(ctx.exception))

    def test_forward_rejects_one_dimensional_states(self):
        """A flat (N,) tensor is rejected instead of crashing on indexing."""
        with self.assertRaises(ValueError):
            self.rbm(torch.rand(5))

    def test_objective_inherits_validation(self):
        """objective() rejects wrong-width batches through forward()."""
        with self.assertRaises(ValueError):
            self.rbm.objective(torch.rand(5, 4), torch.rand(5, 5))

    def test_get_hidden_wrong_width_fails_fast(self):
        """get_hidden rejects wrong visible widths with a clear message."""
        for bad in (torch.rand(5, 4), torch.rand(5, 2)):
            with self.subTest(width=bad.shape[1]):
                with self.assertRaises(ValueError) as ctx:
                    self.rbm.get_hidden(bad)
                self.assertIn("(batch, 3)", str(ctx.exception))

    def test_get_visible_wrong_width_fails_fast(self):
        """get_visible rejects wrong hidden widths with a clear message."""
        with self.assertRaises(ValueError) as ctx:
            self.rbm.get_visible(torch.rand(5, 3))
        self.assertIn("(batch, 2)", str(ctx.exception))


class TestBMStateWidth(unittest.TestCase):
    """State-width validation on the BoltzmannMachine methods."""

    def setUp(self):
        self.bm = BoltzmannMachine(4)

    def test_valid_widths_still_work(self):
        """Correct widths on every method keep working."""
        states = torch.rand(6, 4).round()
        self.assertTrue(torch.isfinite(self.bm(states)).all())
        samples = self.bm.gibbs_sample(num_steps=2, s_visible=states[:, :2])
        self.assertEqual(tuple(samples.shape), (6, 4))

    def test_forward_wrong_width_fails_fast(self):
        """forward rejects wrong-width states with the expected shape."""
        for bad in (torch.rand(5, 3), torch.rand(5, 6)):
            with self.subTest(width=bad.shape[1]):
                with self.assertRaises(ValueError) as ctx:
                    self.bm(bad)
                self.assertIn("(batch, 4)", str(ctx.exception))

    def test_gibbs_sample_rejects_oversized_visible(self):
        """Clamping more units than the model has fails with a clear message."""
        with self.assertRaises(ValueError) as ctx:
            self.bm.gibbs_sample(num_steps=2, s_visible=torch.rand(3, 6))
        self.assertIn("at most 4 columns", str(ctx.exception))

    def test_gibbs_sample_allows_partial_clamping(self):
        """Clamping fewer units than num_nodes stays allowed."""
        samples = self.bm.gibbs_sample(num_steps=2, s_visible=torch.rand(3, 2))
        self.assertEqual(tuple(samples.shape), (3, 4))

    def test_condition_sample_rejects_oversized_visible(self):
        """condition_sample rejects wider visible states up front."""
        with self.assertRaises(ValueError) as ctx:
            self.bm.condition_sample(None, torch.rand(2, 6))
        self.assertIn("at most 4 columns", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
