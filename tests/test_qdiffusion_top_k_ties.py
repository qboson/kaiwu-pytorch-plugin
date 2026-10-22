"""Top-k filtering must keep ties at the cutoff value."""

import torch

from kaiwu.torch_plugin._qdiffusion_sampling import top_k_top_p_filtering


def test_top_k_one_keeps_every_tied_maximum():
    logits = torch.tensor([[1.0, 1.0, 0.0]])

    filtered = top_k_top_p_filtering(logits, top_k=1, top_p=1.0)

    assert filtered.tolist() == [[1.0, 1.0, float("-inf")]]


def test_top_k_two_with_distinct_scores_keeps_two():
    logits = torch.tensor([[3.0, 2.0, 1.0]])

    filtered = top_k_top_p_filtering(logits, top_k=2, top_p=1.0)

    assert filtered.tolist() == [[3.0, 2.0, float("-inf")]]
