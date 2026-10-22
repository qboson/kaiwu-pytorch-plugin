"""Pin the cdf/pdf consistency of the truncated exponential."""

import torch

from kaiwu.torch_plugin.qvae_dist_util import Exponential


def test_cdf_derivative_matches_the_density() -> None:
    grid = torch.linspace(0.05, 0.95, 2001, dtype=torch.float64)
    for beta in (0.5, 1.0, 2.0):
        dist = Exponential(beta)
        cdf = dist.cdf(grid).double()
        density = dist.pdf(grid).double()
        numeric = torch.gradient(cdf, spacing=(grid,))[0]
        assert float((numeric - density).abs().max()) < 5e-3
