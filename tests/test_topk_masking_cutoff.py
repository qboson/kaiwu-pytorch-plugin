"""topk_masking must validate the per-sample cutoff indices."""

import pytest
import torch

from kaiwu.torch_plugin._qdiffusion_sampling import topk_masking

SCORES = torch.randn(2, 5)


def test_one_dimensional_cutoff_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"cutoff_len must have shape \(2, k\)"):
        topk_masking(SCORES, torch.tensor([1, 2]))


def test_mismatched_batch_rows_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"cutoff_len must have shape \(2, k\)"):
        topk_masking(SCORES, torch.tensor([[1], [2], [3]]))


def test_float_cutoff_is_rejected() -> None:
    with pytest.raises(TypeError, match="cutoff_len must be an integer tensor"):
        topk_masking(SCORES, torch.tensor([[1.0], [2.0]]))


@pytest.mark.parametrize("value", [-1, 5])
def test_out_of_range_cutoff_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="must be between 0 and 4"):
        topk_masking(SCORES, torch.tensor([[value], [0]]))


def test_valid_cutoff_returns_a_boolean_mask() -> None:
    mask = topk_masking(SCORES, torch.tensor([[1], [2]]))
    assert mask.shape == SCORES.shape
    assert mask.dtype == torch.bool
