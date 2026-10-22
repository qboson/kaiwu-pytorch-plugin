"""topk_masking must keep exactly cutoff_len lowest-score positions."""

import torch

from kaiwu.torch_plugin._qdiffusion_sampling import topk_masking


def test_lowest_scores_below_the_cutoff_are_marked():
    scores = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

    assert topk_masking(scores, torch.tensor([[2]])).tolist() == [[True, True, False, False]]
    assert topk_masking(scores, torch.tensor([[0]])).tolist() == [[False, False, False, False]]
    assert topk_masking(scores, torch.tensor([[3]])).tolist() == [[True, True, True, False]]


def test_cutoffs_are_per_sample():
    scores = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    mask = topk_masking(scores, torch.tensor([[1], [2]]))

    assert mask.tolist() == [[True, False, False], [True, True, False]]
