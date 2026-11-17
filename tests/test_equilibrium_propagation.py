"""Equilibrium propagation energy, Ising matrix, relaxation and gradient tests."""

import itertools
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import kaiwu.torch_plugin as plugin  # pylint: disable=wrong-import-position
from kaiwu.torch_plugin import (  # pylint: disable=wrong-import-position
    EPConv2d,
    EPLinear,
    EquilibriumPropagation,
)


class EnumerateSampler:  # pylint: disable=too-few-public-methods
    """Returns every spin row so the minimum is exact.

    With ``flip_gauge`` each row is negated, so the auxiliary spin is ``-1``
    and the caller must undo the gauge before reading the spins.
    """

    def __init__(self, flip_gauge=False):
        self.flip_gauge = flip_gauge

    def solve(self, ising_mat):
        """Returns all ``2^n`` assignments with the auxiliary spin last."""
        n_spins = ising_mat.shape[0] - 1
        rows = np.array(
            [list(row) + [1.0] for row in itertools.product([-1.0, 1.0], repeat=n_spins)]
        )
        return -rows if self.flip_gauge else rows


def randomize(model, seed=0):
    """Give every parameter a non-zero float64 value."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=torch.float64))
    return model


def linear_model():
    """A 3-3-2 float64 network of linear layers."""
    return randomize(EquilibriumPropagation(layer_sizes=(3, 3, 2), dtype=torch.float64))


def conv_model():
    """Input to two linear spins, a conv over them, then a linear output."""
    layers = [
        EPLinear(2, 4, dtype=torch.float64),
        EPConv2d((1, 2, 2), out_channels=2, kernel_size=2, dtype=torch.float64),
        EPLinear(2, 2, dtype=torch.float64),
    ]
    return randomize(EquilibriumPropagation(layers), seed=3)


def nudged_energy(model, inputs, states, target, beta):
    """Returns ``E + beta * 1/2 ||s^L - y||^2`` per row."""
    cost = 0.5 * ((states[-1] - target) ** 2).sum(-1)
    return model.energy(inputs, states) + beta * cost


def all_states(model):
    """Yields every ``+-1`` assignment of the non-input layers, batch size 1."""
    widths = list(model.layer_sizes[1:])
    for spins in itertools.product([-1.0, 1.0], repeat=sum(widths)):
        flat = torch.tensor([spins], dtype=torch.float64)
        yield list(torch.split(flat, widths, dim=1))


def brute_force_minimum(model, inputs, target, beta):
    """Returns the exact minimiser of the nudged energy for one row."""
    return min(
        all_states(model),
        key=lambda states: float(nudged_energy(model, inputs, states, target, beta)),
    )


class TestEquilibriumPropagation(unittest.TestCase):
    """Energy, Ising mapping and the EP objective."""

    def assert_ising_matches_energy(self, model, inputs, target, beta):
        """``-z^T J z`` equals the nudged energy up to one constant."""
        ising = model.ising_matrix(inputs[0], target[0], beta)
        np.testing.assert_allclose(ising, ising.T)
        gaps = []
        for states in all_states(model):
            spins = torch.cat(states, dim=1).reshape(-1).numpy()
            full = np.append(spins, 1.0)
            hamiltonian = -full @ ising @ full
            energy = float(nudged_energy(model, inputs, states, target, beta))
            gaps.append(energy - hamiltonian)
        np.testing.assert_allclose(gaps, gaps[0], atol=1e-9)

    def test_ising_matrix_matches_energy(self):
        """The free and nudged Ising objectives match ``E + beta * C``."""
        model = linear_model()
        inputs = torch.tensor([[0.3, -0.7, 0.5]], dtype=torch.float64)
        target = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
        for beta in (0.0, 0.7):
            self.assert_ising_matches_energy(model, inputs, target, beta)

    def test_conv_ising_matrix_matches_energy(self):
        """A convolution between two spin layers maps to the same identity."""
        model = conv_model()
        inputs = torch.tensor([[0.4, -0.2]], dtype=torch.float64)
        target = torch.tensor([[-1.0, 1.0]], dtype=torch.float64)
        self.assert_ising_matches_energy(model, inputs, target, 1.3)

    def test_conv_coupling_matches_drive(self):
        """The dense coupling reproduces ``F.conv2d`` with stride and padding."""
        layer = randomize(
            EPConv2d((2, 4, 4), out_channels=3, kernel_size=3, stride=2, padding=1,
                     dtype=torch.float64)
        )
        spins = torch.sign(torch.randn(5, layer.in_features, dtype=torch.float64))
        torch.testing.assert_close(spins @ layer.coupling(), layer.drive(spins))
        self.assertEqual(layer.out_shape, (3, 2, 2))
        self.assertEqual(tuple(layer.bias_vector().shape), (layer.out_features,))

    def test_equilibrate_returns_exact_minimiser(self):
        """With every row available the chosen spins minimise each phase."""
        model = conv_model()
        inputs = torch.tensor([[0.4, -0.2], [-0.9, 0.6]], dtype=torch.float64)
        target = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64)
        for beta in (0.0, 1.5):
            states = model.equilibrate(inputs, EnumerateSampler(), target, beta)
            self.assertEqual([state.shape[1] for state in states], [4, 2, 2])
            for row in range(inputs.shape[0]):
                best = brute_force_minimum(model, inputs[row : row + 1], target[row], beta)
                for state, expected in zip(states, best):
                    torch.testing.assert_close(state[row : row + 1], expected)

    def test_equilibrate_undoes_the_auxiliary_gauge(self):
        """Rows whose auxiliary spin is ``-1`` decode to the same spins."""
        model = linear_model()
        inputs = torch.tensor([[0.3, -0.7, 0.5]], dtype=torch.float64)
        plain = model.equilibrate(inputs, EnumerateSampler())
        flipped = model.equilibrate(inputs, EnumerateSampler(flip_gauge=True))
        for left, right in zip(plain, flipped):
            torch.testing.assert_close(left, right)

    def test_objective_gradient_is_correlation_difference(self):
        """``dL/dW^l = -(s_n^{l-1} s_n^l - s_f^{l-1} s_f^l) / beta``."""
        model = linear_model()
        beta = 0.5
        inputs = torch.tensor([[0.3, -0.7, 0.5]], dtype=torch.float64)
        free = [torch.tensor([[1.0, -1.0, 1.0]]), torch.tensor([[-1.0, 1.0]])]
        nudged = [torch.tensor([[1.0, 1.0, -1.0]]), torch.tensor([[1.0, -1.0]])]
        free = [state.double() for state in free]
        nudged = [state.double() for state in nudged]
        model.objective(inputs, free, nudged, beta).backward()
        previous_free, previous_nudged = inputs[0], inputs[0]
        for layer, free_state, nudged_state in zip(model.layers, free, nudged):
            expected = -(
                torch.outer(previous_nudged, nudged_state[0])
                - torch.outer(previous_free, free_state[0])
            ) / beta
            torch.testing.assert_close(layer.weight.grad, expected)
            torch.testing.assert_close(
                layer.bias.grad, -(nudged_state[0] - free_state[0]) / beta
            )
            previous_free, previous_nudged = free_state[0], nudged_state[0]

    def test_training_step_updates_parameters(self):
        """One SGD step on the EP objective changes the parameters."""
        model = linear_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        inputs = torch.tensor([[0.3, -0.7, 0.5], [0.9, 0.1, -0.4]], dtype=torch.float64)
        target = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        free = model.equilibrate(inputs, EnumerateSampler())
        nudged = model.equilibrate(inputs, EnumerateSampler(), target, beta=4.0)
        optimizer.zero_grad()
        loss = model.objective(inputs, free, nudged, beta=4.0)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
        )

    def test_invalid_arguments_raise(self):
        """Bad configurations, missing targets and bad sampler output fail early."""
        with self.assertRaises(ValueError):
            EquilibriumPropagation(layer_sizes=(3,))
        with self.assertRaises(ValueError):
            EquilibriumPropagation()
        with self.assertRaises(ValueError):
            EquilibriumPropagation([EPLinear(3, 4), EPLinear(5, 2)])
        with self.assertRaises(ValueError):
            EPConv2d((1, 2, 2), out_channels=1, kernel_size=3)
        model = linear_model()
        inputs = torch.zeros((1, 3), dtype=torch.float64)
        with self.assertRaises(ValueError):
            model.ising_matrix(inputs[0], beta=1.0)
        with self.assertRaises(ValueError):
            model.objective(inputs, [], [], beta=0.0)

        class ShortSampler:  # pylint: disable=too-few-public-methods
            """Returns rows one spin too short."""

            def solve(self, ising_mat):
                """Drops the auxiliary spin."""
                return np.ones((1, ising_mat.shape[0] - 1))

        with self.assertRaises(RuntimeError):
            model.equilibrate(inputs, ShortSampler())

    def test_public_exports(self):
        """The layers and the network are exported at the top level."""
        for name in ("EquilibriumPropagation", "EPLinear", "EPConv2d"):
            self.assertIn(name, plugin.__all__)


def _has_kaiwu_annealer():
    """Whether the installed Kaiwu SDK provides the classical annealer."""
    try:
        from kaiwu.classical import (  # pylint: disable=import-outside-toplevel,unused-import
            SimulatedAnnealingOptimizer,
        )
    except ImportError:
        return False
    return True


@unittest.skipUnless(_has_kaiwu_annealer(), "Kaiwu simulated annealing is not installed")
class TestKaiwuAnnealer(unittest.TestCase):
    """The real SDK annealer on a problem small enough to check exactly."""

    def test_annealer_finds_the_minimum(self):
        """Kaiwu SA reaches the brute-force minimiser of both phases."""
        from kaiwu.classical import (  # pylint: disable=import-outside-toplevel
            SimulatedAnnealingOptimizer,
        )

        model = linear_model()
        sampler = SimulatedAnnealingOptimizer(
            initial_temperature=5.0,
            alpha=0.95,
            cutoff_temperature=0.001,
            iterations_per_t=20,
            size_limit=8,
            rand_seed=7,
            process_num=1,
        )
        inputs = torch.tensor([[0.3, -0.7, 0.5]], dtype=torch.float64)
        target = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
        for beta in (0.0, 1.0):
            states = model.equilibrate(inputs, sampler, target, beta)
            best = brute_force_minimum(model, inputs, target[0], beta)
            energy = float(nudged_energy(model, inputs, states, target, beta))
            expected = float(nudged_energy(model, inputs, best, target, beta))
            self.assertAlmostEqual(energy, expected, places=9)


if __name__ == "__main__":
    unittest.main()
