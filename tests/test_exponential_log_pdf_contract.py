"""log_pdf must be the logarithm of pdf on the support."""

import torch

from kaiwu.torch_plugin.qvae_dist_util import Exponential


def test_log_pdf_is_log_pdf_consistent() -> None:
    grid = torch.linspace(0.05, 0.95, 5)
    for beta in (0.5, 1.3):
        dist = Exponential(beta)
        torch.testing.assert_close(
            dist.log_pdf(grid), torch.log(dist.pdf(grid))
        )
