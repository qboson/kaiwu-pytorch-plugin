"""Pin the normalisation of the truncated exponential density."""

import torch

from kaiwu.torch_plugin.qvae_dist_util import Exponential


def test_pdf_integrates_to_one_on_the_unit_interval() -> None:
    grid = torch.linspace(0.0, 1.0, 20001)
    for beta in (0.1, 1.0, 5.0):
        density = Exponential(beta).pdf(grid)
        integral = torch.trapz(density, grid)
        assert abs(float(integral) - 1.0) < 1e-3
