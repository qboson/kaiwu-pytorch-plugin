"""FeatureSelectionWrapper must validate mask_update_epochs."""

import pytest
import torch

from kaiwu.torch_plugin.maifs import FeatureSelectionWrapper


def _wrapper(**kwargs) -> FeatureSelectionWrapper:
    return FeatureSelectionWrapper(torch.nn.Linear(4, 1), feature_dim=4, **kwargs)


@pytest.mark.parametrize("value", [0, -2])
def test_non_positive_interval_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="mask_update_epochs must be a positive integer"):
        _wrapper(mask_update_epochs=value)


@pytest.mark.parametrize("value", [1.5, True, "2"])
def test_non_integer_interval_is_rejected(value) -> None:
    with pytest.raises(TypeError, match="mask_update_epochs must be a positive integer"):
        _wrapper(mask_update_epochs=value)


def test_valid_interval_is_stored() -> None:
    assert _wrapper(mask_update_epochs=3).mask_update_epochs == 3
    assert _wrapper().mask_update_epochs is None
