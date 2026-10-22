"""Pin the binary entropy formula of the factorial Bernoulli utility."""

import torch

from kaiwu.torch_plugin.qvae_dist_util import FactorialBernoulliUtil


def test_entropy_matches_the_binary_entropy() -> None:
    logits = torch.tensor([[-2.0, 0.5, 3.0], [1.0, -1.0, 0.0]])
    mu = torch.sigmoid(logits)

    expected = -(mu * torch.log(mu) + (1 - mu) * torch.log(1 - mu))
    torch.testing.assert_close(FactorialBernoulliUtil(logits).entropy(), expected)
