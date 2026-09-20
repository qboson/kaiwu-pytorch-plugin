"""Regression coverage for nonlinear gradients and integration edge cases."""
import functools
import unittest

import torch

from qdeq import QDEQ
from qdeq.diagnostics import power_method
from qdeq.solvers import anderson_solver


class TestIntegrationRegressions(unittest.TestCase):
    def test_nonsymmetric_nonlinear_gradients_match_unrolling(self):
        torch.manual_seed(21)
        weight = (torch.randn(3, 3, dtype=torch.float64) * 0.08).requires_grad_()
        x = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
        target = torch.randn_like(x)
        f = lambda z, u: torch.tanh(z @ weight.T + u)
        solver = functools.partial(anderson_solver, lam=1e-10)
        deq = QDEQ(f_solver=solver, b_solver=solver, f_eps=1e-10, b_eps=1e-10)
        out, info = deq(f, torch.zeros_like(x), x)
        actual = torch.autograd.grad((out * target).sum(), (weight, x))
        z = torch.zeros_like(x)
        for _ in range(100):
            z = f(z, x)
        expected = torch.autograd.grad((z * target).sum(), (weight, x))
        torch.testing.assert_close(out, z, atol=1e-8, rtol=1e-8)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-7)
        self.assertEqual(info["eps"], 1e-10)
        self.assertEqual(info["backward_eps"], 1e-10)

    def test_state_independent_map_has_zero_jacobian(self):
        x = torch.randn(2, 3, requires_grad=True)
        out, _ = QDEQ(f_solver=anderson_solver)(lambda z, u: 2*u, torch.zeros_like(x), x)
        out.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 2))

    def test_anderson_accepts_batch_feature_shapes(self):
        for shape in [(2, 3), (2, 3, 4), (2, 2, 3, 4)]:
            x = torch.ones(shape)
            result = anderson_solver(lambda z: 0.3*z+x, torch.zeros_like(x), eps=1e-6)
            torch.testing.assert_close(result["result"], x/0.7, atol=1e-5, rtol=1e-5)

    def test_anderson_rejects_unusable_budgets(self):
        for kwargs in [dict(threshold=2), dict(m=1), dict(stop_mode="invalid")]:
            with self.assertRaises(ValueError):
                anderson_solver(lambda z: z, torch.zeros(1, 2), **kwargs)

    def test_tolerance_defaults_are_not_overridden(self):
        seen = []
        def solve(f, z, **kwargs):
            seen.append(kwargs)
            return anderson_solver(f, z, **kwargs)
        x = torch.ones(1, 2, requires_grad=True)
        QDEQ(f_solver=solve, b_solver=solve)(lambda z: 0.2*z+x, torch.zeros_like(x))[0].sum().backward()
        self.assertEqual(seen, [{"stop_mode": "rel"}, {"stop_mode": "rel"}])

    def test_power_diagnostic_preserves_training_graph(self):
        weight = torch.tensor(0.3, requires_grad=True)
        x = torch.ones(1, 2, 3, requires_grad=True)
        out, info = QDEQ(f_solver=anderson_solver)(lambda z: weight*z+x, torch.zeros_like(x))
        _, radius = power_method(info["new_z"], info["z_star"], n_iters=5)
        torch.testing.assert_close(radius, torch.tensor([[0.3]]))
        out.sum().backward()
        self.assertTrue(torch.isfinite(weight.grad))
        torch.testing.assert_close(x.grad, torch.full_like(x, 1/0.7), atol=1e-4, rtol=1e-4)

    def test_zero_jacobian_power_diagnostic_is_finite(self):
        z = torch.ones(1, 2, requires_grad=True)
        vec, radius = power_method(z*0, z, n_iters=3)
        self.assertTrue(torch.isfinite(vec).all())
        torch.testing.assert_close(radius, torch.zeros(1, 1))
