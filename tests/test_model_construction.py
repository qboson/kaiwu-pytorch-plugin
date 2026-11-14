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

from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine


class TestRestrictedBoltzmannMachineConstruction(unittest.TestCase):
    """Constructor validation for RestrictedBoltzmannMachine."""

    def test_plain_int_sizes(self):
        """Plain positive integers construct as before."""
        model = RestrictedBoltzmannMachine(3, 2)
        self.assertEqual(model.num_nodes, 5)

    def test_numpy_integer_sizes(self):
        """numpy integers from array shapes are accepted as sizes."""
        shape = np.zeros((4, 6)).shape
        model = RestrictedBoltzmannMachine(shape[0], shape[1])
        self.assertEqual(model.num_nodes, 10)

    def test_supplied_tensors_are_preserved(self):
        """Correctly shaped supplied parameters are used unchanged."""
        generator = torch.Generator().manual_seed(31)
        weights = torch.randn((3, 2), generator=generator)
        bias = torch.randn((5,), generator=generator)
        model = RestrictedBoltzmannMachine(3, 2, weights, bias)
        self.assertTrue(torch.equal(model.quadratic_coef.detach(), weights))
        self.assertTrue(torch.equal(model.linear_bias.detach(), bias))
        states = torch.rand(7, 5).round()
        self.assertTrue(torch.isfinite(model(states)).all())

    def test_non_positive_sizes_fail_fast(self):
        """Zero and negative sizes are rejected with the argument name."""
        for bad in (0, -1):
            with self.subTest(num_visible=bad):
                with self.assertRaises(ValueError) as ctx:
                    RestrictedBoltzmannMachine(bad, 2)
                self.assertIn("num_visible", str(ctx.exception))

    def test_fractional_and_non_numeric_sizes_fail_fast(self):
        """Fractional floats and None are rejected instead of crashing later."""
        for bad in (2.5, None, [3]):
            with self.subTest(num_visible=bad):
                with self.assertRaises(ValueError):
                    RestrictedBoltzmannMachine(bad, 2)

    def test_wrong_shape_weights_fail_at_construction(self):
        """A mis-shaped quadratic_coef is rejected immediately, not at train time."""
        with self.assertRaises(ValueError) as ctx:
            RestrictedBoltzmannMachine(3, 2, quadratic_coef=torch.randn(4, 2))
        self.assertIn("quadratic_coef", str(ctx.exception))
        self.assertIn("(3, 2)", str(ctx.exception))

    def test_wrong_length_bias_fail_at_construction(self):
        """A mis-shaped linear_bias is rejected immediately."""
        with self.assertRaises(ValueError) as ctx:
            RestrictedBoltzmannMachine(3, 2, linear_bias=torch.randn(6))
        self.assertIn("linear_bias", str(ctx.exception))
        self.assertIn("(5,)", str(ctx.exception))

    def test_non_tensor_parameters_fail_with_hint(self):
        """Lists and arrays are rejected with a conversion hint."""
        for bad in ([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], np.ones((3, 2))):
            with self.subTest(value=type(bad).__name__):
                with self.assertRaises(TypeError) as ctx:
                    RestrictedBoltzmannMachine(3, 2, quadratic_coef=bad)
                self.assertIn("torch.as_tensor", str(ctx.exception))


class TestBoltzmannMachineConstruction(unittest.TestCase):
    """Constructor validation for BoltzmannMachine."""

    def test_plain_int_sizes(self):
        """Plain positive integers construct as before."""
        model = BoltzmannMachine(4)
        self.assertEqual(model.num_nodes, 4)

    def test_numpy_integer_sizes(self):
        """numpy integers are accepted as sizes."""
        model = BoltzmannMachine(np.int64(4))
        self.assertEqual(model.num_nodes, 4)

    def test_supplied_tensors_are_preserved(self):
        """Correctly shaped supplied parameters are used unchanged."""
        generator = torch.Generator().manual_seed(32)
        weights = torch.randn((4, 4), generator=generator)
        bias = torch.randn((4,), generator=generator)
        model = BoltzmannMachine(4, weights, bias)
        self.assertTrue(torch.equal(model.quadratic_coef.detach(), weights))
        self.assertTrue(torch.equal(model.linear_bias.detach(), bias))
        states = torch.rand(7, 4).round()
        self.assertTrue(torch.isfinite(model(states)).all())

    def test_non_positive_sizes_fail_fast(self):
        """Zero and negative sizes are rejected with the argument name."""
        for bad in (0, -3):
            with self.subTest(num_nodes=bad):
                with self.assertRaises(ValueError) as ctx:
                    BoltzmannMachine(bad)
                self.assertIn("num_nodes", str(ctx.exception))

    def test_fractional_and_non_numeric_sizes_fail_fast(self):
        """Fractional floats and None are rejected with a clear error."""
        for bad in (4.5, None):
            with self.subTest(num_nodes=bad):
                with self.assertRaises(ValueError):
                    BoltzmannMachine(bad)

    def test_non_square_weights_fail_at_construction(self):
        """A non-square quadratic_coef is rejected immediately."""
        with self.assertRaises(ValueError) as ctx:
            BoltzmannMachine(4, quadratic_coef=torch.randn(4, 5))
        self.assertIn("quadratic_coef", str(ctx.exception))
        self.assertIn("(4, 4)", str(ctx.exception))

    def test_wrong_length_bias_fail_at_construction(self):
        """A mis-shaped linear_bias is rejected immediately."""
        with self.assertRaises(ValueError) as ctx:
            BoltzmannMachine(4, linear_bias=torch.randn(5))
        self.assertIn("linear_bias", str(ctx.exception))

    def test_non_tensor_parameters_fail_with_hint(self):
        """Arrays are rejected with a conversion hint."""
        with self.assertRaises(TypeError) as ctx:
            BoltzmannMachine(4, quadratic_coef=np.ones((4, 4)))
        self.assertIn("torch.as_tensor", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
