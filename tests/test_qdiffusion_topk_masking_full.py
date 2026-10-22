"""A cutoff equal to the sequence length keeps every non-maximum score."""

import torch

from kaiwu.torch_plugin._qdiffusion_sampling import topk_masking


def test_the_largest_cutoff_marks_all_but_the_maximum():
    scores = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

    mask = topk_masking(scores, torch.tensor([[3]]))

    assert mask.tolist() == [[True, True, True, False]]
