import unittest

import torch

from qdeq.solvers import anderson_solver


class TestAndersonSolver(unittest.TestCase):
    def test_finds_analytic_fixed_point_of_linear_contraction(self):
        torch.manual_seed(0)
        bsz, d, L = 2, 4, 1
        weight = torch.eye(d) * 0.3
        bias = torch.randn(1, d, 1)

        def f(z):
            return torch.einsum("ij,bjl->bil", weight, z) + bias

        # Analytic fixed point: z* = (I - W)^{-1} b
        analytic = torch.einsum(
            "ij,bjl->bil", torch.inverse(torch.eye(d) - weight), bias.expand(bsz, d, L)
        )

        z0 = torch.zeros(bsz, d, L)
        out = anderson_solver(f, z0, threshold=40, eps=1e-6, stop_mode="rel")

        self.assertIn("result", out)
        self.assertIn("nstep", out)
        torch.testing.assert_close(out["result"], analytic, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
