"""top_k_top_p_filtering must restore the original tensor shape."""

import torch

from kaiwu.torch_plugin._qdiffusion_sampling import top_k_top_p_filtering


def test_three_dimensional_input_keeps_its_shape():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 4)

    filtered = top_k_top_p_filtering(logits, top_k=2, top_p=1.0)

    assert filtered.shape == logits.shape
    kept = (filtered != float("-inf")).sum(dim=-1)
    assert kept.tolist() == [[2, 2, 2], [2, 2, 2]]


def test_three_dimensional_input_without_filtering_is_unchanged():
    logits = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    filtered = top_k_top_p_filtering(logits, top_k=0, top_p=1.0)

    torch.testing.assert_close(filtered, logits)
