"""FeatureSelectionWrapper must validate feature_dim at construction."""

import pytest
import torch

from kaiwu.torch_plugin.maifs import FeatureSelectionWrapper


def _wrapper(feature_dim) -> FeatureSelectionWrapper:
    return FeatureSelectionWrapper(torch.nn.Linear(4, 1), feature_dim=feature_dim)


@pytest.mark.parametrize("feature_dim", [0, -4])
def test_non_positive_feature_dim_is_rejected(feature_dim) -> None:
    with pytest.raises(ValueError, match="feature_dim must be a positive integer"):
        _wrapper(feature_dim)


@pytest.mark.parametrize("feature_dim", [4.0, True, "4"])
def test_non_integer_feature_dim_is_rejected(feature_dim) -> None:
    with pytest.raises(TypeError, match="feature_dim must be a positive integer"):
        _wrapper(feature_dim)


def test_valid_feature_dim_builds_the_mask() -> None:
    selector = _wrapper(4)
    assert selector.feature_dim == 4
    assert selector.mask.shape == (4,)
