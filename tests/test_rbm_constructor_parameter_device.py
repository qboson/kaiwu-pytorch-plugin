"""Caller-supplied RBM parameters must follow the constructor device.

``RestrictedBoltzmannMachine`` accepts pretrained-ish coefficient tensors in
the constructor and a ``device`` argument. Before this regression, supplied
tensors kept their original device while ``self.device`` switched to the
requested one, so on a CUDA host ``sample()``/``get_ising_matrix()`` mixed CPU
parameters with CUDA scratch tensors and failed. The shipped
``example/run_rbm.py`` passes CPU tensors without a device and therefore broke
on every CUDA host.

The meta-device case runs on CPU-only machines; the CUDA case is skipped when
no accelerator is available.
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin import RestrictedBoltzmannMachine


class RBMConstructorParameterDeviceTests(unittest.TestCase):
    """Constructor device contract for caller-supplied RBM parameters."""

    def setUp(self):
        self.weights = torch.tensor(
            [[0.5, -1.0], [2.0, 0.25], [-0.5, 1.5]], dtype=torch.float32
        )
        self.bias = torch.tensor([0.1, -0.2, 0.3, 0.4, -0.5], dtype=torch.float32)

    def _model(self, device):
        return RestrictedBoltzmannMachine(
            num_visible=3,
            num_hidden=2,
            quadratic_coef=self.weights,
            linear_bias=self.bias,
            device=device,
        )

    def test_supplied_parameters_follow_requested_device(self):
        """Parameters live on the requested device and inputs stay untouched."""
        model = self._model(torch.device("meta"))

        self.assertEqual(model.quadratic_coef.device, torch.device("meta"))
        self.assertEqual(model.linear_bias.device, torch.device("meta"))
        self.assertEqual(torch.device(model.device), torch.device("meta"))
        self.assertEqual(self.weights.device, torch.device("cpu"))
        self.assertEqual(self.bias.device, torch.device("cpu"))

    def test_supplied_parameters_keep_values_dtype_and_gradients(self):
        """Moving to the requested device must not copy values or break autograd."""
        weights = self.weights.double()
        bias = self.bias.double()
        model = RestrictedBoltzmannMachine(
            num_visible=3,
            num_hidden=2,
            quadratic_coef=weights,
            linear_bias=bias,
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(model.quadratic_coef.detach(), weights)
        torch.testing.assert_close(model.linear_bias.detach(), bias)
        self.assertEqual(model.quadratic_coef.dtype, torch.float64)
        self.assertEqual(model.linear_bias.dtype, torch.float64)

        states = torch.ones(2, 5, dtype=torch.float64)
        model(states).sum().backward()
        self.assertIsNotNone(model.quadratic_coef.grad)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_model_with_cpu_parameters_builds_ising_matrix(self):
        """The shipped CUDA path must accept CPU-supplied parameters."""
        model = self._model(torch.device("cuda"))

        self.assertEqual(model.quadratic_coef.device.type, "cuda")
        self.assertEqual(model.linear_bias.device.type, "cuda")
        ising_matrix = model.get_ising_matrix()
        self.assertEqual(ising_matrix.shape, (6, 6))


if __name__ == "__main__":
    unittest.main()
