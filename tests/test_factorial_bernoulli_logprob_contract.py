"""Pin the factorial Bernoulli log-probability formula."""

import torch

from kaiwu.torch_plugin.qvae_dist_util import FactorialBernoulliUtil


def test_log_prob_matches_the_cross_entropy_formula() -> None:
    logits = torch.tensor([[-1.0, 0.5, 2.0], [0.0, -2.0, 1.0]])
    samples = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])

    expected = -(
        logits - logits * samples + torch.nn.functional.softplus(-logits)
    )
    torch.testing.assert_close(
        FactorialBernoulliUtil(logits).log_prob_per_var(samples), expected
    )


def test_all_zero_samples_give_negative_softplus() -> None:
    logits = torch.tensor([[-1.0, 0.0, 2.0]])
    samples = torch.zeros(1, 3)
    expected = -torch.nn.functional.softplus(logits)
    torch.testing.assert_close(
        FactorialBernoulliUtil(logits).log_prob_per_var(samples), expected
    )
