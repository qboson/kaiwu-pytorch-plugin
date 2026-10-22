"""Pin the mask accessors of FeatureSelectionWrapper."""

import numpy as np
import torch
from torch import nn

from kaiwu.torch_plugin.maifs import FeatureSelectionWrapper


def _selector() -> FeatureSelectionWrapper:
    selector = FeatureSelectionWrapper(nn.Linear(5, 1), feature_dim=5)
    with torch.no_grad():
        selector.mask.copy_(torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0]))
    return selector


def test_support_is_boolean_and_matches_the_mask() -> None:
    support = _selector().get_support()
    assert support.dtype == np.bool_
    assert support.tolist() == [True, False, True, True, False]


def test_selected_indices_are_sorted_flat_indices() -> None:
    assert _selector().selected_indices().tolist() == [0, 2, 3]


def test_num_selected_counts_selected_features() -> None:
    assert _selector().num_selected() == 3
