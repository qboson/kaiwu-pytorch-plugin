"""The feature-count projection must enforce the minimum as well."""

import numpy as np
from torch import nn

from kaiwu.torch_plugin import FeatureSelectionWrapper


def _wrapper(min_selected_features=None):
    return FeatureSelectionWrapper(
        model=nn.Identity(),
        feature_dim=4,
        min_selected_features=min_selected_features,
    )


def test_an_explicit_minimum_is_enforced():
    wrapper = _wrapper(min_selected_features=2)
    quadratic = np.eye(4)
    linear = -np.array([4.0, 3.0, 2.0, 1.0])

    projected = wrapper._project_selected_feature_count(  # pylint: disable=protected-access
        np.zeros(4, dtype=int), quadratic, linear
    )

    assert projected.sum() == 2
    assert projected.tolist() == [1, 1, 0, 0]


def test_an_empty_mask_reaches_the_default_minimum():
    wrapper = _wrapper()

    projected = wrapper._project_selected_feature_count(  # pylint: disable=protected-access
        np.zeros(4, dtype=int), np.eye(4), -np.arange(4.0, 0.0, -1.0)
    )

    assert projected.sum() == 1
