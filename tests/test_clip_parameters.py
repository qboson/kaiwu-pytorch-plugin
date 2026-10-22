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


class ClipParametersMixin:
    """Shared expectations for clip_parameters on RBM and BoltzmannMachine.

    Not a TestCase itself: the concrete classes below mix it in with
    ``unittest.TestCase`` so the abstract hooks are never collected.
    """

    def make_model(self):
        """Build the model under test with out-of-range parameters."""
        raise NotImplementedError

    def test_linear_bias_is_clipped_by_h_range(self):
        """h_range bounds the linear bias (local fields), per the name."""
        model = self.make_model()
        model.linear_bias.data.fill_(5.0)
        model.clip_parameters(h_range=(-0.1, 0.1), j_range=(-2.0, 2.0))
        self.assertLessEqual(model.linear_bias.data.max().item(), 0.1 + 1e-7)
        self.assertGreaterEqual(model.linear_bias.data.min().item(), -0.1 - 1e-7)

    def test_quadratic_coef_is_clipped_by_j_range(self):
        """j_range bounds the quadratic weights (couplings), per the name."""
        model = self.make_model()
        model.quadratic_coef.data.fill_(5.0)
        model.clip_parameters(h_range=(-0.1, 0.1), j_range=(-2.0, 2.0))
        self.assertLessEqual(model.quadratic_coef.data.max().item(), 2.0 + 1e-7)
        self.assertGreaterEqual(model.quadratic_coef.data.min().item(), -2.0 - 1e-7)

    def test_clip_is_in_place_and_returns_none(self):
        """The call mutates the existing parameters and returns nothing."""
        model = self.make_model()
        bias_before = model.linear_bias
        result = model.clip_parameters(h_range=(-1.0, 1.0), j_range=(-1.0, 1.0))
        self.assertIsNone(result)
        self.assertIs(model.linear_bias, bias_before)

    def test_values_inside_range_are_unchanged(self):
        """Parameters already inside the ranges keep their exact values."""
        model = self.make_model()
        model.linear_bias.data = torch.full_like(model.linear_bias.data, 0.25)
        model.clip_parameters(h_range=(-1.0, 1.0), j_range=(-1.0, 1.0))
        self.assertTrue(
            torch.equal(model.linear_bias.data, torch.full_like(model.linear_bias.data, 0.25))
        )

    def test_equal_bounds_are_allowed(self):
        """A degenerate (low == high) range pins values instead of failing."""
        model = self.make_model()
        model.linear_bias.data = torch.randn_like(model.linear_bias.data)
        model.clip_parameters(h_range=(0.5, 0.5), j_range=(-1.0, 1.0))
        self.assertTrue(
            torch.allclose(model.linear_bias.data, torch.full_like(model.linear_bias.data, 0.5))
        )

    def test_inverted_h_range_raises(self):
        """An inverted h_range fails fast instead of collapsing the bias.

        Before validation, ``clamp_(1.0, -1.0)`` silently mapped every
        linear-bias value to -1.0 and destroyed the trained weights.
        """
        model = self.make_model()
        with self.assertRaises(ValueError):
            model.clip_parameters(h_range=(1.0, -1.0), j_range=(-1.0, 1.0))

    def test_inverted_j_range_raises(self):
        """An inverted j_range fails fast instead of collapsing the weights."""
        model = self.make_model()
        with self.assertRaises(ValueError):
            model.clip_parameters(h_range=(-1.0, 1.0), j_range=(1.0, -1.0))

    def test_non_pair_range_raises(self):
        """A scalar or longer sequence is rejected with a clear error."""
        model = self.make_model()
        with self.assertRaises(ValueError):
            model.clip_parameters(h_range=0.5, j_range=(-1.0, 1.0))
        with self.assertRaises(ValueError):
            model.clip_parameters(h_range=(-1.0, 1.0), j_range=(-1.0, 0.0, 1.0))

    def test_parameters_are_untouched_after_validation_error(self):
        """A rejected call must not modify the model on its way out."""
        model = self.make_model()
        bias_before = model.linear_bias.data.clone()
        weights_before = model.quadratic_coef.data.clone()
        with self.assertRaises(ValueError):
            model.clip_parameters(h_range=(1.0, -1.0), j_range=(-1.0, 1.0))
        self.assertTrue(torch.equal(model.linear_bias.data, bias_before))
        self.assertTrue(torch.equal(model.quadratic_coef.data, weights_before))


class TestRestrictedBoltzmannMachineClip(ClipParametersMixin, unittest.TestCase):
    """clip_parameters semantics for RestrictedBoltzmannMachine."""

    def make_model(self):
        """Build a small RBM with fixed sizes."""
        return RestrictedBoltzmannMachine(num_visible=4, num_hidden=2)


class TestBoltzmannMachineClip(ClipParametersMixin, unittest.TestCase):
    """clip_parameters semantics for BoltzmannMachine."""

    def make_model(self):
        """Build a small fully connected BM."""
        return BoltzmannMachine(num_nodes=4)


if __name__ == "__main__":
    unittest.main()
