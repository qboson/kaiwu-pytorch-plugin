import unittest

import torch

from qdeq.diagnostics import jac_loss_estimate, power_method


class TestDiagnostics(unittest.TestCase):
    def test_jac_loss_estimate_matches_known_linear_map(self):
        torch.manual_seed(0)
        bsz, d = 4, 8
        weight = torch.randn(d, d) * 0.1
        z0 = torch.randn(bsz, d, requires_grad=True)
        f0 = z0 @ weight.T

        estimate = jac_loss_estimate(f0, z0, vecs=64, create_graph=False)
        exact = (weight.T @ weight).trace() / d

        torch.testing.assert_close(estimate, exact, atol=0.05, rtol=0.05)

    def test_power_method_finds_dominant_eigenvalue(self):
        torch.manual_seed(0)
        bsz, d = 1, 6
        weight = torch.diag(torch.tensor([0.9, 0.5, 0.3, 0.2, 0.1, 0.05]))
        z0 = torch.randn(bsz, d, requires_grad=True)
        f0 = z0 @ weight.T

        _evector, evalue = power_method(f0, z0, n_iters=100)
        torch.testing.assert_close(evalue, torch.tensor([[0.9]]), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
