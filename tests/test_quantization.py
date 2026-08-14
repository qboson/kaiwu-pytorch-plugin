import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn

from kaiwu.torch_plugin import quantization


class QuantizationTests(unittest.TestCase):
    def test_activation_and_hessian_with_sa(self):
        model = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(18, 1),
        ).eval()
        x = torch.randn(4, 1, 3, 3)
        solvers = [{"name": "sa", "params": {"iterations_per_t": 10, "size_limit": 1, "rand_seed": 1}}]

        _, activation_report = quantization(model, bits=3, solvers=solvers).activation(x)
        self.assertEqual(activation_report["method"], "activation")
        self.assertEqual([layer["name"] for layer in activation_report["layers"]], ["0", "2"])

        hessians = {
            "0": torch.eye(model[0].weight.numel()).tolist(),
            "2": torch.eye(model[2].weight.numel()).tolist(),
        }
        _, hessian_report = quantization(model, bits=3, solvers=solvers).hessian(hessians)
        self.assertEqual(hessian_report["method"], "hessian")
        self.assertEqual([layer["name"] for layer in hessian_report["layers"]], ["0", "2"])

        _, forward_report = quantization(model, bits=3, solvers=solvers)(x, method="activation")
        self.assertEqual(forward_report["method"], "activation")

    def test_selected_layers(self):
        model = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(18, 1),
        ).eval()
        x = torch.randn(4, 1, 3, 3)
        solvers = [{"name": "sa", "params": {"iterations_per_t": 10, "size_limit": 1, "rand_seed": 1}}]

        _, activation_report = quantization(model, bits=3, solvers=solvers, layers="2").activation(x)
        self.assertEqual(activation_report["target_layers"], ["2"])
        self.assertEqual([layer["name"] for layer in activation_report["layers"]], ["2"])

        hessians = {"2": torch.eye(model[2].weight.numel()).tolist()}
        _, hessian_report = quantization(model, bits=3, solvers=solvers, layers=["2"]).hessian(hessians)
        self.assertEqual([layer["name"] for layer in hessian_report["layers"]], ["2"])


if __name__ == "__main__":
    unittest.main()
