"""Energy, gradient, and continuous-relaxation tests."""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from kaiwu.torch_plugin import EPConfig, EquilibriumPropagation


class TestEquilibriumPropagation(unittest.TestCase):
    """Check the public energy API without a Kaiwu install."""

    def test_parameter_gradient_matches_the_correlation(self):
        """dE/dW^l = -s^l (s^{l+1})^T and dE/db^l = -s^l."""
        model = EquilibriumPropagation(EPConfig(layer_sizes=(2, 2, 2), seed=1))
        features = torch.tensor([0.2, 0.8], dtype=torch.float64)
        hidden = torch.tensor([0.3, 0.7], dtype=torch.float64)
        output = torch.tensor([0.4, 0.6], dtype=torch.float64)
        model.energy(features, [hidden, output]).backward()
        torch.testing.assert_close(model.weights[0].grad, -torch.outer(features, hidden))
        torch.testing.assert_close(model.weights[1].grad, -torch.outer(hidden, output))
        torch.testing.assert_close(model.biases[0].grad, -hidden)
        torch.testing.assert_close(model.biases[1].grad, -output)

    def test_continuous_trace_does_not_increase(self):
        """Projected steps keep F_beta non-increasing."""
        model = EquilibriumPropagation(
            EPConfig(layer_sizes=(2, 2, 2), seed=2, step_size=0.1, max_steps=8)
        )
        features = torch.tensor([0.25, 0.75], dtype=torch.float64)
        target = torch.tensor([1.0, 0.0], dtype=torch.float64)
        result = model.equilibrate(features, target, beta=0.5, backend="continuous")
        trace = result["energy_trace"]
        self.assertGreaterEqual(len(trace), 2)
        self.assertIn(result["stop_reason"], {"converged", "max_steps", "energy_increase"})
        for previous, current in zip(trace, trace[1:]):
            tolerance = 1e-12 * max(1.0, abs(previous))
            self.assertLessEqual(current, previous + tolerance)

    def test_kaiwu_annealer_matches_the_qubo_objective(self):
        """The default backend decodes a Kaiwu spin row back into F_beta."""
        model = EquilibriumPropagation(
            EPConfig(
                layer_sizes=(2, 2),
                seed=11,
                iterations_per_t=20,
                candidate_count=16,
            )
        )
        features = torch.tensor([0.25, 0.75], dtype=torch.float64)
        target = torch.tensor([1.0, 0.0], dtype=torch.float64)
        result = model.equilibrate(features, target, beta=0.5)
        self.assertEqual(result["stop_reason"], "budget_exhausted")
        self.assertEqual(result["n_bits"], 2)
        for state in result["states"]:
            self.assertTrue(torch.all((state == 0.0) | (state == 1.0)))
        self.assertAlmostEqual(
            float(result["qubo_objective"]), float(result["energy"]), delta=1e-8
        )
        again = model.equilibrate(features, target, beta=0.5)
        for first, second in zip(result["states"], again["states"]):
            torch.testing.assert_close(first, second)

        bits = torch.cat(result["states"]).detach().numpy()
        spins = np.concatenate([np.where(bits > 0.5, 1.0, -1.0), [1.0]])
        warmed = model.equilibrate(features, target, beta=0.5, init_solution=spins)
        self.assertEqual(warmed["n_bits"], 2)
        self.assertAlmostEqual(float(warmed["qubo_objective"]), float(warmed["energy"]), delta=1e-8)

    def test_positive_beta_requires_a_target(self):
        """A nudged relaxation without y is a configuration error."""
        model = EquilibriumPropagation(EPConfig(layer_sizes=(2, 2), seed=0))
        features = torch.tensor([0.2, 0.8], dtype=torch.float64)
        with self.assertRaises(ValueError):
            model.equilibrate(features, beta=1.0)

    def test_config_rejects_invalid_shapes_and_round_trips(self):
        """Layer widths and bit width are checked, and dict conversion is exact."""
        with self.assertRaises(ValueError):
            EPConfig(layer_sizes=(2,))
        with self.assertRaises(ValueError):
            EPConfig(layer_sizes=(2, 0))
        with self.assertRaises(ValueError):
            EPConfig(layer_sizes=(2, 2), state_bits=0)
        config = EPConfig(layer_sizes=(4, 3, 2), state_bits=2, beta=0.25, seed=7)
        self.assertEqual(EPConfig.from_dict(config.to_dict()), config)


if __name__ == "__main__":
    unittest.main()
