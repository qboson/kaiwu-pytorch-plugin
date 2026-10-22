"""QDEQ must work with the real Kaiwu BruteForceOptimizer (no license)."""

import torch
from kaiwu.classical import BruteForceOptimizer

from kaiwu.torch_plugin import QDEQ, QDEQConfig


class LinearLayer(torch.nn.Module):
    """``layer(z, x) = W z + x`` with spectral norm 0.3."""

    def __init__(self, dim=4, seed=0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        weight = torch.randn(dim, dim, generator=generator, dtype=torch.float64)
        self.weight = torch.nn.Parameter(
            0.3 * weight / torch.linalg.matrix_norm(weight, 2)
        )

    def forward(self, z, x):
        """Returns ``W z + x``."""
        return z @ self.weight.T + x


def test_qdeq_converges_with_the_sdk_brute_force_sampler():
    config = QDEQConfig(
        f_thres=40,
        f_eps=1e-6,
        num_basis=1,
        coeff_bits=2,
        coeff_range=2.0,
    )
    layer = LinearLayer()
    x = torch.randn(2, 4, dtype=torch.float64)
    deq = QDEQ(layer, BruteForceOptimizer(), config).eval()

    out = deq(x)

    eye = torch.eye(4, dtype=torch.float64)
    exact = torch.linalg.solve(eye - layer.weight.detach(), x.T).T
    assert deq.get_last_stats()["forward_residual"] < 1e-4
    torch.testing.assert_close(out, exact, rtol=1e-4, atol=1e-4)
