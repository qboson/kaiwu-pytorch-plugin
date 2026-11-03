"""Training-step tests for equilibrium propagation."""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from kaiwu.torch_plugin import EPConfig, EquilibriumPropagation


def _model(layer_sizes=(2, 2, 2), **kwargs):
    """Return a tiny float64 network with a non-zero coupling."""
    config = EPConfig(layer_sizes=layer_sizes, state_bits=1, seed=0, **kwargs)
    model = EquilibriumPropagation(config)
    with torch.no_grad():
        for weight in model.weights:
            weight.copy_(torch.linspace(-0.4, 0.5, weight.numel(), dtype=torch.float64).reshape_as(weight))
        for bias in model.biases:
            bias.copy_(torch.linspace(-0.2, 0.3, bias.numel(), dtype=torch.float64))
    return model


class TestEPTrain(unittest.TestCase):
    """Check the two-phase update without relying on a long anneal."""

    def test_train_batch_changes_parameters(self):
        """One exhaustive batch takes a single SGD step."""
        model = _model()
        before = [value.detach().clone() for value in model.parameters()]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        features = torch.tensor([[0.2, 0.8], [0.6, 0.1]], dtype=torch.float64)
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        stats = model.train_batch(
            features, targets, optimizer, beta=1.0, backend="exhaustive", epoch=0
        )
        self.assertEqual(stats["batch_size"], 2)
        self.assertEqual(stats["solver_calls"], 4)
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))

    def test_autograd_matches_the_correlation_difference(self):
        """The loss gradient is the free-minus-nudged outer product over beta."""
        model = _model()
        features = torch.tensor([0.25, 0.75], dtype=torch.float64)
        target = torch.tensor([1.0, 0.0], dtype=torch.float64)
        beta = 1.0
        free = model.equilibrate(features, target, beta=0.0, backend="exhaustive", seed=4)
        nudged = model.equilibrate(
            features,
            target,
            beta=beta,
            backend="exhaustive",
            seed=4,
            init_solution=free["init_solution"],
        )
        self.assertEqual(free["init_solution"].shape, (5,))
        self.assertEqual(float(free["init_solution"][-1]), 1.0)
        free_states = [state.detach() for state in free["states"]]
        nudged_states = [state.detach() for state in nudged["states"]]
        loss = (model.energy(features, nudged_states) - model.energy(features, free_states)) / beta
        loss.backward()
        hidden_free, output_free = free_states
        hidden_nudged, output_nudged = nudged_states
        torch.testing.assert_close(
            model.weights[0].grad,
            (torch.outer(features, hidden_free) - torch.outer(features, hidden_nudged)) / beta,
        )
        torch.testing.assert_close(
            model.weights[1].grad,
            (torch.outer(hidden_free, output_free) - torch.outer(hidden_nudged, output_nudged)) / beta,
        )
        torch.testing.assert_close(model.biases[0].grad, (hidden_free - hidden_nudged) / beta)
        torch.testing.assert_close(model.biases[1].grad, (output_free - output_nudged) / beta)

    def test_non_positive_beta_is_rejected_before_solving(self):
        """A zero nudge makes the EP gradient undefined."""
        model = _model(layer_sizes=(2, 2))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        features = torch.tensor([[0.2, 0.8]], dtype=torch.float64)
        targets = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        with self.assertRaises(ValueError):
            model.train_batch(features, targets, optimizer, beta=0.0, backend="exhaustive")

    def test_predict_keeps_parameters_and_returns_class_indices(self):
        """Prediction is a free phase and does not train."""
        model = _model()
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        features = torch.tensor([[0.2, 0.8], [0.9, 0.1]], dtype=torch.float64)
        predicted = model.predict(features, backend="exhaustive", seed=1)
        self.assertEqual(tuple(predicted.shape), (2,))
        self.assertEqual(predicted.dtype, torch.long)
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, before[name])

    def test_kaiwu_train_batch_is_reproducible(self):
        """Two identical SA batches land on the same parameters."""
        features = torch.tensor([[0.2, 0.8]], dtype=torch.float64)
        targets = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        first = _train_once(features, targets)
        second = _train_once(features, targets)
        self.assertEqual(first["solver_calls"], second["solver_calls"])
        for left, right in zip(first["parameters"], second["parameters"]):
            torch.testing.assert_close(left, right)


def _train_once(features, targets):
    """Train one SA step from a fixed initialisation."""
    model = EquilibriumPropagation(
        EPConfig(layer_sizes=(2, 2), state_bits=1, seed=5, iterations_per_t=8, candidate_count=4)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    stats = model.train_batch(features, targets, optimizer, beta=0.5, backend="sa", epoch=1)
    stats["parameters"] = [value.detach().clone() for value in model.parameters()]
    return stats


if __name__ == "__main__":
    unittest.main()
