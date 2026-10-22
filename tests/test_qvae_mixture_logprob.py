"""MixtureGeneric.log_prob_per_var must equal the mixture density."""

import torch

from kaiwu.torch_plugin.qvae_dist_util import MixtureGeneric


def test_log_prob_matches_the_mixture_density():
    param = torch.tensor([-1.0, 0.0, 1.0])
    dist = MixtureGeneric(param, smoothing_dist_beta=0.5)
    samples = torch.tensor([0.2, 0.5, 0.9])

    log_prob = dist.log_prob_per_var(samples)

    q = torch.sigmoid(param)
    pdf_0 = dist.smoothing_dist.pdf(samples)
    pdf_1 = dist.smoothing_dist.pdf(1.0 - samples)
    expected = torch.log(q * pdf_1 + (1.0 - q) * pdf_0)
    assert log_prob.shape == samples.shape
    torch.testing.assert_close(log_prob, expected)
