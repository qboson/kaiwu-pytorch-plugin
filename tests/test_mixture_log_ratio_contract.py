"""Pin the mixture log-ratio used by the KL gradient."""

import torch

from kaiwu.torch_plugin.qvae_dist_util import MixtureGeneric


def test_log_ratio_is_log_pdf_one_minus_z_minus_log_pdf_z() -> None:
    mixture = MixtureGeneric(torch.zeros(2, 3), 1.0)
    zeta = torch.tensor([[0.2, 0.8, 0.5], [0.1, 0.9, 0.3]])

    expected = mixture.smoothing_dist.log_pdf(1.0 - zeta) - mixture.smoothing_dist.log_pdf(
        zeta
    )
    torch.testing.assert_close(mixture.log_ratio(zeta), expected)
