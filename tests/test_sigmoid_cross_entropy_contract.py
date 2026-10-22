"""Pin the sigmoid cross-entropy helper used by the Bernoulli utilities."""

import torch

from kaiwu.torch_plugin.qvae_dist_util import sigmoid_cross_entropy_with_logits


def test_formula_matches_the_definition() -> None:
    logits = torch.tensor([[-2.0, 0.0, 1.5], [3.0, -0.5, 0.25]])
    labels = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

    expected = logits - logits * labels + torch.nn.functional.softplus(-logits)
    torch.testing.assert_close(sigmoid_cross_entropy_with_logits(logits, labels), expected)


def test_zero_labels_reduce_to_softplus() -> None:
    logits = torch.tensor([-1.0, 0.0, 2.0])
    expected = torch.nn.functional.softplus(logits)
    torch.testing.assert_close(
        sigmoid_cross_entropy_with_logits(logits, torch.zeros(3)), expected
    )
