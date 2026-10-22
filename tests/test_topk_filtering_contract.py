"""Pin the top-k retention of top_k_top_p_filtering."""

import math

import torch

from kaiwu.torch_plugin._qdiffusion_sampling import top_k_top_p_filtering


def test_top_k_one_keeps_only_the_maximum() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 1.0]])

    filtered = top_k_top_p_filtering(logits, top_k=1, top_p=1.0)

    finite = torch.isfinite(filtered)
    assert finite.sum(dim=1).tolist() == [1, 1]
    assert torch.all(filtered[0][1] == 3.0)
    assert torch.all(filtered[1][0] == 4.0)
    assert torch.all(filtered[~finite] == -math.inf)


def test_top_k_zero_disables_top_k_filtering() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    filtered = top_k_top_p_filtering(logits, top_k=0, top_p=1.0)
    torch.testing.assert_close(filtered, logits)
