"""QDEQ public interface, solver and gradient tests with Kaiwu simulated annealing."""

import unittest

import torch

import kaiwu.torch_plugin as plugin
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import QDEQ, QDEQConfig


def kaiwu_sa(seed=0):
    """Kaiwu simulated annealing with default settings and a fixed seed."""
    return SimulatedAnnealingOptimizer(rand_seed=seed)


class LinearLayer(torch.nn.Module):
    """layer(z, x) = W z + x with spectral norm 0.5, so z* = (I - W)^-1 x."""

    def __init__(self, dim=6, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        weight = torch.randn(dim, dim, generator=gen, dtype=torch.float64)
        self.weight = torch.nn.Parameter(0.5 * weight / torch.linalg.matrix_norm(weight, 2))

    def forward(self, z, x):
        """Returns ``W z + x``."""
        return z @ self.weight.T + x


def solve_config(**kwargs):
    """Tight tolerances so results can be compared with the analytic solution."""
    defaults = {"f_thres": 40, "f_eps": 1e-6, "b_thres": 80, "b_eps": 1e-11}
    defaults.update(kwargs)
    return QDEQConfig(**defaults)


class TestQDEQConstruction(unittest.TestCase):
    """Exports and argument validation."""

    def test_exported_names(self):
        """QDEQ and QDEQConfig are exported by the package."""
        self.assertIn("QDEQ", plugin.__all__)
        self.assertIn("QDEQConfig", plugin.__all__)

    def test_forward_requires_sampler(self):
        """Solving the fixed point without a sampler raises."""
        with self.assertRaises(ValueError):
            QDEQ(LinearLayer(), None)(torch.zeros(1, 6, dtype=torch.float64))

    def test_rejects_unknown_stop_mode(self):
        """Unknown stop-mode names are rejected at construction."""
        with self.assertRaises(ValueError):
            QDEQ(LinearLayer(), kaiwu_sa(), QDEQConfig(stop_mode="max"))


class TestIsingForward(unittest.TestCase):
    """The Ising forward solver reaches the analytic fixed point."""

    def setUp(self):
        self.layer = LinearLayer()
        self.x = torch.randn(3, 6, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
        eye = torch.eye(6, dtype=torch.float64)
        self.exact = torch.linalg.solve(eye - self.layer.weight.detach(), self.x.T).T

    def test_converges_to_fixed_point(self):
        """Kaiwu SA steps reach (I - W)^-1 x."""
        deq = QDEQ(self.layer, kaiwu_sa(), solve_config()).eval()
        out = deq(self.x)
        stats = deq.get_last_stats()
        torch.testing.assert_close(out, self.exact, rtol=1e-5, atol=1e-5)
        self.assertLess(stats["forward_residual"], 1e-6)
        self.assertGreater(stats["forward_sampler_time"], 0.0)


class TestImplicitGradient(unittest.TestCase):
    """Gradients equal the implicit-function-theorem gradient (I - W^T)^-1 g."""

    def setUp(self):
        self.layer = LinearLayer()
        self.x = torch.randn(3, 6, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
        self.x.requires_grad_()
        self.weight = torch.linspace(-1, 1, 18, dtype=torch.float64).view(3, 6)
        eye = torch.eye(6, dtype=torch.float64)
        self.expected = torch.linalg.solve(eye - self.layer.weight.detach().T, self.weight.T).T

    def test_gradient_matches_implicit_function_theorem(self):
        """Input and parameter gradients come from the implicit backward pass."""
        deq = QDEQ(self.layer, kaiwu_sa(), solve_config())
        (deq(self.x) * self.weight).sum().backward()
        torch.testing.assert_close(self.x.grad, self.expected, rtol=1e-5, atol=1e-5)
        self.assertIsNotNone(self.layer.weight.grad)
        self.assertGreater(deq.get_last_stats()["backward_nstep"], 0)

    def test_eval_mode_builds_no_implicit_graph(self):
        """Evaluation without gradients attaches no backward hook."""
        deq = QDEQ(self.layer, kaiwu_sa(), solve_config()).eval()
        with torch.no_grad():
            out = deq(self.x)
        self.assertFalse(out.requires_grad)
        self.assertNotIn("backward_nstep", deq.get_last_stats())

    def test_jacobian_loss_keeps_implicit_gradient(self):
        """Adding the regulariser leaves the implicit gradient of the output unchanged."""
        deq = QDEQ(self.layer, kaiwu_sa(), solve_config(jac_loss_vecs=2))
        out = deq(self.x)
        (out.sum() + deq.last_jac_loss).backward()
        eye = torch.eye(6, dtype=torch.float64)
        ones = torch.ones(6, 3, dtype=torch.float64)
        expected = torch.linalg.solve(eye - self.layer.weight.detach().T, ones).T
        torch.testing.assert_close(self.x.grad, expected, rtol=1e-5, atol=1e-5)


class TestHooksAndRegulariser(unittest.TestCase):
    """Subclass contexts and the Jacobian regulariser."""

    def test_contexts_wrap_solve_and_reattach(self):
        """solve_context wraps every no-grad call and reattach_context the final call."""
        events = []

        class Layer(torch.nn.Module):
            """Contraction that records whether each call tracks gradients."""

            def forward(self, z, x):
                """Returns ``z / 2 + x``."""
                events.append("grad" if torch.is_grad_enabled() else "nograd")
                return 0.5 * z + x

        class Recorder:
            """Context manager that records entry and exit."""

            def __init__(self, name):
                self.name = name

            def __enter__(self):
                events.append(self.name + ":enter")

            def __exit__(self, *exc):
                events.append(self.name + ":exit")

        class HookedQDEQ(QDEQ):
            """QDEQ with recording contexts."""

            def solve_context(self):
                """Records the solve phase."""
                return Recorder("solve")

            def reattach_context(self):
                """Records the final layer call."""
                return Recorder("reattach")

        HookedQDEQ(Layer(), kaiwu_sa(), QDEQConfig(f_thres=4))(torch.randn(2, 4))
        self.assertEqual(events[0], "solve:enter")
        self.assertEqual(events[-3:], ["reattach:enter", "grad", "reattach:exit"])
        self.assertNotIn("grad", events[:-3])

    def test_jacobian_loss_estimate(self):
        """For J = I / 2 the estimate is close to 0.25."""
        z = torch.randn(4, 64, dtype=torch.float64, requires_grad=True)
        self.assertAlmostEqual(QDEQ.jacobian_loss(0.5 * z, z, vecs=50).item(), 0.25, delta=0.02)


class TestCustomForwardSolver(unittest.TestCase):
    """Subclasses can replace the forward solver and keep the implicit backward pass."""

    def test_override_solve_fixed_point(self):
        """A plain-iteration subclass needs no sampler and gets the implicit gradient."""

        class IteratedQDEQ(QDEQ):
            """QDEQ whose forward fixed point is found by plain iteration."""

            def solve_fixed_point(self, func, z0):
                """Iterates ``z <- func(z)`` a fixed number of times."""
                z = z0
                for _ in range(200):
                    z = func(z)
                return z, {"nstep": 200}

        layer = LinearLayer()
        x = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
        deq = IteratedQDEQ(layer, None, solve_config())
        out = deq(x)
        eye = torch.eye(6, dtype=torch.float64)
        exact = torch.linalg.solve(eye - layer.weight.detach(), x.detach().T).T
        torch.testing.assert_close(out, exact, rtol=1e-8, atol=1e-8)

        out.sum().backward()
        ones = torch.ones(6, 3, dtype=torch.float64)
        expected = torch.linalg.solve(eye - layer.weight.detach().T, ones).T
        torch.testing.assert_close(x.grad, expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(deq.get_last_stats()["forward_nstep"], 200)


if __name__ == "__main__":
    unittest.main()
