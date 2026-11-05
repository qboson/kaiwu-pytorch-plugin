"""FeatureSelectionWrapper must validate the feature-count bounds it is given."""

import pytest
import torch

from kaiwu.torch_plugin.maifs import FeatureSelectionWrapper


def _wrapper(**kwargs) -> FeatureSelectionWrapper:
    return FeatureSelectionWrapper(
        model=torch.nn.Linear(4, 1),
        feature_dim=4,
        **kwargs,
    )


def test_min_above_max_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not exceed max_selected_features"):
        _wrapper(min_selected_features=3, max_selected_features=2)


def test_negative_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        _wrapper(min_selected_features=-1)
    with pytest.raises(ValueError, match="must be non-negative"):
        _wrapper(max_selected_features=-2)


def test_bounds_above_feature_dim_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not exceed feature_dim"):
        _wrapper(max_selected_features=5)
    with pytest.raises(ValueError, match="must not exceed feature_dim"):
        _wrapper(min_selected_features=6, max_selected_features=6)


def test_valid_bounds_are_stored() -> None:
    selector = _wrapper(min_selected_features=2, max_selected_features=3)
    assert selector.min_selected_features == 2
    assert selector.max_selected_features == 3

    default_selector = _wrapper()
    assert 0 <= default_selector.min_selected_features
    assert default_selector.min_selected_features <= 4
