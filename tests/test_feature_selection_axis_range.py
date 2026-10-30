"""input_feature_axis must be inside the input tensor's rank."""

from __future__ import annotations

import pytest
import torch

from kaiwu.torch_plugin import FeatureSelectionWrapper

FEATURE_DIM = 4


def _selector(axis: int) -> FeatureSelectionWrapper:
    return FeatureSelectionWrapper(
        torch.nn.Linear(FEATURE_DIM, 1),
        feature_dim=FEATURE_DIM,
        input_feature_axis=axis,
        min_selected_features=1,
        max_selected_features=3,
    )


@pytest.mark.parametrize("axis", [2, 3, -3])
def test_out_of_range_axis_is_rejected_for_two_dimensional_inputs(axis: int) -> None:
    with pytest.raises(ValueError, match="input_feature_axis"):
        _selector(axis).apply_mask(torch.rand(2, FEATURE_DIM))


def test_out_of_range_axis_is_rejected_for_feature_first_inputs() -> None:
    with pytest.raises(ValueError, match="input_feature_axis"):
        _selector(2).apply_mask(torch.rand(FEATURE_DIM, 2))


@pytest.mark.parametrize("axis", [0, 1, -1, -2])
def test_in_range_axes_are_accepted(axis: int) -> None:
    inputs = (
        torch.rand(FEATURE_DIM, 2) if axis in (0, -2) else torch.rand(2, FEATURE_DIM)
    )
    masked = _selector(axis).apply_mask(inputs)
    assert masked.shape == inputs.shape
