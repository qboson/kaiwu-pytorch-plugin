"""The projection must validate the candidate mask shape."""

import numpy as np
import pytest
from torch import nn

from kaiwu.torch_plugin import FeatureSelectionWrapper


def test_a_wrong_length_mask_is_rejected():
    wrapper = FeatureSelectionWrapper(model=nn.Identity(), feature_dim=4)

    with pytest.raises(ValueError, match="shape"):
        wrapper._project_selected_feature_count(  # pylint: disable=protected-access
            np.zeros(3, dtype=int), np.eye(4), np.zeros(4)
        )
