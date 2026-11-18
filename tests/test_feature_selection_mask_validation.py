"""Caller-supplied feature masks must be one-dimensional and full width."""

from __future__ import annotations

import pytest
import torch

from kaiwu.torch_plugin import FeatureSelectionWrapper

FEATURE_DIM = 4


def _selector() -> FeatureSelectionWrapper:
    return FeatureSelectionWrapper(
        torch.nn.Linear(FEATURE_DIM, 1),
        feature_dim=FEATURE_DIM,
        min_selected_features=1,
        max_selected_features=3,
    )


def _inputs() -> torch.Tensor:
    return torch.arange(8, dtype=torch.float32).reshape(2, FEATURE_DIM)


@pytest.mark.parametrize("width", [1, FEATURE_DIM - 1, FEATURE_DIM + 1])
def test_wrong_mask_width_is_rejected(width: int) -> None:
    selector = _selector()
    with pytest.raises(ValueError, match="mask must have shape"):
        selector.apply_mask(_inputs(), mask=torch.ones(width))


def test_two_dimensional_mask_is_rejected() -> None:
    selector = _selector()
    with pytest.raises(ValueError, match="mask must have shape"):
        selector.apply_mask(_inputs(), mask=torch.ones(1, FEATURE_DIM))


def test_non_tensor_mask_is_rejected() -> None:
    selector = _selector()
    with pytest.raises(TypeError, match="mask must be a torch.Tensor"):
        selector.apply_mask(_inputs(), mask=[1, 0, 1, 0])


def test_custom_mask_zeroes_the_requested_features() -> None:
    selector = _selector()
    inputs = _inputs()
    masked = selector.apply_mask(inputs, mask=torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert torch.equal(masked[:, [1, 3]], torch.zeros(2, 2))
    assert torch.equal(masked[:, [0, 2]], inputs[:, [0, 2]])


def test_continuous_masks_stay_supported() -> None:
    selector = _selector()
    inputs = _inputs()
    masked = selector.apply_mask(inputs, mask=torch.full((FEATURE_DIM,), 0.5))
    assert torch.equal(masked, inputs * 0.5)


def test_default_mask_is_unchanged() -> None:
    selector = _selector()
    inputs = _inputs()
    assert torch.equal(selector.apply_mask(inputs), inputs)
