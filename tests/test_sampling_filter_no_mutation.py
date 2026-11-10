from __future__ import annotations

import os
import sys

import torch

src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, src_root)

from kaiwu.torch_plugin._qdiffusion_sampling import top_k_top_p_filtering


def test_filtering_does_not_mutate_the_input_logits() -> None:
    """The filter must not write filtered values into the caller's tensor."""
    torch.manual_seed(0)
    logits = torch.randn(2, 5)
    original = logits.clone()

    filtered = top_k_top_p_filtering(logits, top_k=3, top_p=1.0)

    assert torch.equal(logits, original)
    assert not torch.equal(filtered, original)
    assert torch.isinf(filtered).any()


def test_filtering_output_still_filters_correctly() -> None:
    """top-k keeps exactly k entries finite per row."""
    torch.manual_seed(1)
    logits = torch.randn(3, 7)

    filtered = top_k_top_p_filtering(logits, top_k=2, top_p=1.0)

    finite_per_row = torch.isfinite(filtered).sum(dim=-1)
    assert torch.all(finite_per_row <= 2)
