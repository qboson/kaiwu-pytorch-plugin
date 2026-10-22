"""topk_masking keeps only scores strictly below the cutoff value."""

import torch

from kaiwu.torch_plugin._qdiffusion_sampling import topk_masking


def test_ties_at_the_cutoff_are_not_marked():
    scores = torch.tensor([[0.1, 0.2, 0.2, 0.3]])

    mask = topk_masking(scores, torch.tensor([[2]]))

    assert mask.tolist() == [[True, False, False, False]]
