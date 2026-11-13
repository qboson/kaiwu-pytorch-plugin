"""Explicit BM construction devices apply to supplied parameters too.

The meta cases verify device placement and forward compatibility on CPU-only
machines; they do not claim accelerator numerical validation. No sampler,
network/model download, SDK license, or backend execution is needed.
"""

import unittest

import torch

from kaiwu.torch_plugin import BoltzmannMachine


class ConstructorParameterDeviceTests(unittest.TestCase):
    def test_explicit_meta_device_with_supplied_and_default_parameters(self):
        for supplied in ("both", "weights", "bias"):
            with self.subTest(supplied=supplied):
                weights = torch.tensor([[0.0, 2.0], [0.0, 0.0]])
                bias = torch.tensor([1.0, -1.0])
                before_weights = weights.clone()
                before_bias = bias.clone()
                kwargs = {}
                if supplied in ("both", "weights"):
                    kwargs["quadratic_coef"] = weights
                if supplied in ("both", "bias"):
                    kwargs["linear_bias"] = bias

                model = BoltzmannMachine(2, device="meta", **kwargs)

                self.assertEqual(model.quadratic_coef.device, torch.device("meta"))
                self.assertEqual(model.linear_bias.device, torch.device("meta"))
                self.assertEqual(torch.device(model.device), torch.device("meta"))
                self.assertTrue(all(p.is_leaf and p.requires_grad
                                    for p in model.parameters()))
                energies = model(torch.ones(3, 2, device="meta"))
                self.assertEqual(energies.device, torch.device("meta"))
                self.assertEqual(energies.shape, (3,))
                self.assertEqual(weights.device, torch.device("cpu"))
                self.assertEqual(bias.device, torch.device("cpu"))
                torch.testing.assert_close(weights, before_weights)
                torch.testing.assert_close(bias, before_bias)

    def test_supplied_cpu_parameters_keep_values_dtype_and_gradients(self):
        weights = torch.tensor([[0.0, 2.0], [0.0, 0.0]], dtype=torch.float64)
        bias = torch.tensor([1.0, -1.0], dtype=torch.float64)
        model = BoltzmannMachine(2, quadratic_coef=weights, linear_bias=bias,
                                 device=torch.device("cpu"))
        states = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)

        energies = model(states)
        energies.sum().backward()

        self.assertEqual(model.quadratic_coef.device, torch.device("cpu"))
        self.assertEqual(model.linear_bias.device, torch.device("cpu"))
        self.assertEqual(model.quadratic_coef.dtype, torch.float64)
        self.assertEqual(model.linear_bias.dtype, torch.float64)
        torch.testing.assert_close(model.quadratic_coef, weights)
        torch.testing.assert_close(model.linear_bias, bias)
        torch.testing.assert_close(energies, torch.tensor([-2.0, 1.0], dtype=torch.float64))
        torch.testing.assert_close(model.linear_bias.grad,
                                   torch.tensor([-1.0, -2.0], dtype=torch.float64))
        torch.testing.assert_close(model.quadratic_coef.grad,
                                   torch.tensor([[0.0, -1.0], [0.0, 0.0]],
                                                dtype=torch.float64))

    def test_default_parameter_device_control(self):
        model = BoltzmannMachine(2, device="meta")
        self.assertEqual(model.quadratic_coef.device, torch.device("meta"))
        self.assertEqual(model.linear_bias.device, torch.device("meta"))

    def test_none_device_follows_existing_resolved_device(self):
        weights = torch.tensor([[0.0, 2.0], [0.0, 0.0]], dtype=torch.float64)
        bias = torch.tensor([1.0, -1.0], dtype=torch.float64)
        model = BoltzmannMachine(2, quadratic_coef=weights, linear_bias=bias,
                                 device=None)
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.assertEqual(torch.device(model.device), resolved)
        self.assertEqual(model.quadratic_coef.device, resolved)
        self.assertEqual(model.linear_bias.device, resolved)
        self.assertEqual(model.quadratic_coef.dtype, torch.float64)
        self.assertEqual(model.linear_bias.dtype, torch.float64)
        self.assertEqual(weights.device, torch.device("cpu"))
        self.assertEqual(bias.device, torch.device("cpu"))
        torch.testing.assert_close(model.quadratic_coef.detach().cpu(), weights)
        torch.testing.assert_close(model.linear_bias.detach().cpu(), bias)


if __name__ == "__main__":
    unittest.main()
