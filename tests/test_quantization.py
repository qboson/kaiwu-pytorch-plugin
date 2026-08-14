"""Unit tests for the QUBO quantization converter."""

# pylint: disable=wrong-import-position

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn

from kaiwu.torch_plugin import quantization


class QuantizationTests(unittest.TestCase):
    """Exercise activation and Hessian PTQ paths."""

    def test_activation_and_hessian_with_sa(self):
        """Quantize a small model with SA through both PTQ methods."""

        test_model = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(18, 1),
        ).eval()
        input_tensor = torch.randn(4, 1, 3, 3)
        solver_configs = [
            {
                "name": "sa",
                "params": {"iterations_per_t": 10, "size_limit": 1, "rand_seed": 1},
            }
        ]

        _, activation_report = quantization(
            test_model,
            bits=3,
            solvers=solver_configs,
        ).activation(input_tensor)
        self.assertEqual(activation_report["method"], "activation")
        self.assertEqual([layer["name"] for layer in activation_report["layers"]], ["0", "2"])

        hessians = {
            "0": torch.eye(test_model[0].weight.numel()).tolist(),
            "2": torch.eye(test_model[2].weight.numel()).tolist(),
        }
        _, hessian_report = quantization(
            test_model,
            bits=3,
            solvers=solver_configs,
        ).hessian(hessians)
        self.assertEqual(hessian_report["method"], "hessian")
        self.assertEqual([layer["name"] for layer in hessian_report["layers"]], ["0", "2"])

        _, forward_report = quantization(
            test_model,
            bits=3,
            solvers=solver_configs,
        )(input_tensor, method="activation")
        self.assertEqual(forward_report["method"], "activation")

    def test_selected_layers(self):
        """Quantize only user-selected layer names."""

        test_model = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(18, 1),
        ).eval()
        input_tensor = torch.randn(4, 1, 3, 3)
        solver_configs = [
            {
                "name": "sa",
                "params": {"iterations_per_t": 10, "size_limit": 1, "rand_seed": 1},
            }
        ]

        _, activation_report = quantization(
            test_model,
            bits=3,
            solvers=solver_configs,
            layers="2",
        ).activation(input_tensor)
        self.assertEqual(activation_report["target_layers"], ["2"])
        self.assertEqual([layer["name"] for layer in activation_report["layers"]], ["2"])

        hessians = {"2": torch.eye(test_model[2].weight.numel()).tolist()}
        _, hessian_report = quantization(
            test_model,
            bits=3,
            solvers=solver_configs,
            layers=["2"],
        ).hessian(hessians)
        self.assertEqual([layer["name"] for layer in hessian_report["layers"]], ["2"])


if __name__ == "__main__":
    unittest.main()
