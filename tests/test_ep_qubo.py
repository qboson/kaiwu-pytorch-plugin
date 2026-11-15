"""QUBO identity tests for the equilibrium-propagation mapping."""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from kaiwu.torch_plugin import EPConfig, EquilibriumPropagation
from kaiwu.torch_plugin._ep_qubo import decode_bits, encode_states, logical_bit_count


def _model(layer_sizes, state_bits):
    """Return a tiny network with fixed parameters."""
    config = EPConfig(layer_sizes=layer_sizes, state_bits=state_bits, seed=0)
    model = EquilibriumPropagation(config)
    with torch.no_grad():
        for weight in model.weights:
            weight.copy_(torch.linspace(-0.3, 0.4, weight.numel(), dtype=torch.float64).reshape_as(weight))
        for index, bias in enumerate(model.biases):
            bias.copy_(torch.linspace(-0.2, 0.2, bias.numel(), dtype=torch.float64))
            bias.add_(0.05 * index)
    return model


class TestEPQubo(unittest.TestCase):
    """Check bit layout, the energy identity, and the exhaustive minimiser."""

    def test_logical_bit_count_matches_the_digits_network(self):
        """64-32-10 uses 42, 84, and 126 bits at K = 1, 2, 3."""
        self.assertEqual(logical_bit_count((64, 32, 10), 1), 42)
        self.assertEqual(logical_bit_count((64, 32, 10), 2), 84)
        self.assertEqual(logical_bit_count((64, 32, 10), 3), 126)

    def test_bit_encoding_round_trips(self):
        """Every assignment decodes and encodes back to itself."""
        for state_bits in (1, 2):
            layer_sizes = (2, 2, 2)
            n_bits = logical_bit_count(layer_sizes, state_bits)
            for mask in range(1 << n_bits):
                bits = _bits(mask, n_bits)
                restored = encode_states(decode_bits(bits, layer_sizes, state_bits), state_bits)
                self.assertTrue(torch.equal(bits, restored))

    def test_qubo_matches_total_energy_on_every_assignment(self):
        """q^T Q q + offset equals F_beta for hidden and input-output networks."""
        cases = (
            ((2, 2, 2), 1, 0.5),
            ((2, 2, 2), 2, 0.5),
            ((2, 2), 1, 1.0),
            ((2, 2), 2, 0.0),
        )
        features = torch.tensor([0.25, 0.75], dtype=torch.float64)
        target = torch.tensor([1.0, 0.0], dtype=torch.float64)
        for layer_sizes, state_bits, beta in cases:
            model = _model(layer_sizes, state_bits)
            self._assert_identity(model, features, target, beta, state_bits)

    def test_free_phase_qubo_ignores_the_target(self):
        """beta = 0 drops both the nudge and the constant."""
        model = _model((2, 2, 2), 1)
        features = torch.tensor([0.25, 0.75], dtype=torch.float64)
        first, first_offset = model.qubo(features, torch.tensor([1.0, 0.0]), 0.0)
        second, second_offset = model.qubo(features, torch.tensor([0.0, 1.0]), 0.0)
        torch.testing.assert_close(first, second)
        self.assertEqual(float(first_offset), 0.0)
        self.assertEqual(float(second_offset), 0.0)
        self.assertTrue(torch.equal(torch.tril(first, diagonal=-1), torch.zeros_like(first)))

    def test_exhaustive_relaxation_finds_the_minimum(self):
        """The decoded exhaustive state matches a brute-force energy minimum."""
        model = _model((2, 2, 2), 1)
        features = torch.tensor([0.25, 0.75], dtype=torch.float64)
        target = torch.tensor([1.0, 0.0], dtype=torch.float64)
        result = model.equilibrate(features, target, beta=0.5, backend="exhaustive")
        minimum = _minimum_energy(model, features, target, 0.5, 1)
        self.assertAlmostEqual(float(result["energy"]), minimum, delta=1e-9)
        self.assertAlmostEqual(float(result["qubo_objective"]), float(result["energy"]), delta=1e-9)
        self.assertEqual(result["n_bits"], 4)

    def test_exhaustive_backend_rejects_problems_above_its_limit(self):
        """Enumeration stops before building assignments that exceed the cap."""
        config = EPConfig(layer_sizes=(2, 2, 2), state_bits=1, exhaustive_max_bits=3, seed=0)
        model = EquilibriumPropagation(config)
        features = torch.tensor([0.25, 0.75], dtype=torch.float64)
        with self.assertRaises(ValueError):
            model.equilibrate(features, beta=0.0, backend="exhaustive")

    def _assert_identity(self, model, features, target, beta, state_bits):
        matrix, offset = model.qubo(features, target, beta)
        self.assertTrue(torch.equal(torch.tril(matrix, diagonal=-1), torch.zeros_like(matrix)))
        n_bits = int(matrix.shape[0])
        for mask in range(1 << n_bits):
            bits = _bits(mask, n_bits)
            states = decode_bits(bits, model.layer_sizes, state_bits)
            quad = torch.dot(bits, matrix @ bits)
            energy = model.total_energy(features, states, target, beta)
            self.assertAlmostEqual(float(quad + offset), float(energy), delta=1e-9)


def _bits(mask, n_bits):
    """Expand an integer assignment into a float64 bit vector."""
    return torch.tensor([(mask >> index) & 1 for index in range(n_bits)], dtype=torch.float64)


def _minimum_energy(model, features, target, beta, state_bits):
    """Return the smallest F_beta over the full K-bit grid."""
    n_bits = logical_bit_count(model.layer_sizes, state_bits)
    best = None
    for mask in range(1 << n_bits):
        states = decode_bits(_bits(mask, n_bits), model.layer_sizes, state_bits)
        value = float(model.total_energy(features, states, target, beta))
        best = value if best is None else min(best, value)
    return best


if __name__ == "__main__":
    unittest.main()
