"""Feature-count projection must validate and enforce the bounds."""

import numpy as np
import pytest
import torch
from torch import nn

from kaiwu.torch_plugin import FeatureSelectionWrapper


def _wrapper():
    return FeatureSelectionWrapper(
        model=nn.Identity(),
        feature_dim=4,
        min_selected_features=1,
        max_selected_features=2,
    )


def test_a_non_binary_mask_is_rejected():
    with pytest.raises(ValueError, match="binary"):
        _wrapper()._project_selected_feature_count(  # pylint: disable=protected-access
            np.array([1.0, 0.5, 0.0, 1.0]),
            np.eye(4),
            np.zeros(4),
        )


def test_the_maximum_feature_count_is_enforced():
    wrapper = _wrapper()
    quadratic = np.eye(4)
    linear = -np.array([3.0, 2.0, 1.0, 0.5])

    projected = wrapper._project_selected_feature_count(  # pylint: disable=protected-access
        np.array([1, 1, 1, 0]), quadratic, linear
    )

    assert projected.sum() == 2
    assert projected.tolist() in ([1, 1, 0, 0], [1, 0, 1, 0])
