"""top_k_top_p_filtering must validate its filtering parameters."""

import pytest
import torch

from kaiwu.torch_plugin._qdiffusion_sampling import top_k_top_p_filtering

LOGITS = torch.randn(2, 6)


@pytest.mark.parametrize("top_k", [-1, -5])
def test_negative_top_k_is_rejected(top_k) -> None:
    with pytest.raises(ValueError, match="top_k must be a non-negative integer"):
        top_k_top_p_filtering(LOGITS, top_k=top_k)


@pytest.mark.parametrize("top_k", [1.5, True])
def test_non_integer_top_k_is_rejected(top_k) -> None:
    with pytest.raises(TypeError, match="top_k must be a non-negative integer"):
        top_k_top_p_filtering(LOGITS, top_k=top_k)


@pytest.mark.parametrize("top_p", [0.0, -0.5, 1.5, float("nan"), float("inf")])
def test_top_p_outside_the_probability_range_is_rejected(top_p) -> None:
    with pytest.raises(ValueError, match=r"top_p must be in \(0, 1\]"):
        top_k_top_p_filtering(LOGITS, top_p=top_p)


def test_valid_parameters_still_filter() -> None:
    filtered = top_k_top_p_filtering(LOGITS, top_k=2, top_p=0.9)
    assert filtered.shape == LOGITS.shape
    assert top_k_top_p_filtering(LOGITS, top_k=0, top_p=1.0).shape == LOGITS.shape
