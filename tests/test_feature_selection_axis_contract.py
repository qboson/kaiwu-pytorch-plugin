"""Pin the feature-axis handling of FeatureSelectionWrapper.apply_mask."""

import torch
from torch import nn

from kaiwu.torch_plugin.maifs import FeatureSelectionWrapper


def _selector(**kwargs) -> FeatureSelectionWrapper:
    return FeatureSelectionWrapper(nn.Linear(3, 1), feature_dim=3, **kwargs)


def test_mask_is_applied_along_the_configured_axis() -> None:
    selector = _selector(input_feature_axis=1)
    inputs = torch.ones(2, 3, 4)
    mask = torch.tensor([1.0, 0.0, 1.0])

    masked = selector.apply_mask(inputs, mask=mask)

    expected = inputs * mask.reshape(1, 3, 1)
    torch.testing.assert_close(masked, expected)


def test_default_axis_is_the_last_dimension() -> None:
    selector = _selector()
    inputs = torch.ones(2, 4, 3)
    mask = torch.tensor([0.0, 1.0, 1.0])

    masked = selector.apply_mask(inputs, mask=mask)

    expected = inputs * mask.reshape(1, 1, 3)
    torch.testing.assert_close(masked, expected)
