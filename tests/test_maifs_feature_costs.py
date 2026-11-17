"""Feature-specific acquisition costs contribute to the MAIFS binary objective."""

import numpy as np
import pytest
import torch
from torch import nn

from kaiwu.torch_plugin import FeatureSelectionWrapper


def test_expensive_feature_can_be_dropped_with_the_builtin_solver():
    costs = np.array([0.0, 1.0])
    model = nn.Linear(2, 1, bias=False)
    nn.init.ones_(model.weight)
    selector = FeatureSelectionWrapper(model, 2, feature_costs=costs)
    costs.fill(5.0)

    # Each feature has the same effect on MSE. Only feature 1 costs more than
    # the corresponding improvement in prediction error.
    data = [(torch.eye(2), torch.ones(2, 1))]
    np.testing.assert_array_equal(selector.update_mask(data, nn.MSELoss()), [1, 0])
    np.testing.assert_array_equal(selector.get_support(), [True, False])


@pytest.mark.parametrize("costs", [[1.0], [[1.0, 0.0]], [1.0, np.inf], [-0.1, 1.0]])
def test_invalid_feature_costs_are_rejected(costs):
    with pytest.raises(ValueError, match="one finite nonnegative cost per feature"):
        FeatureSelectionWrapper(nn.Linear(2, 1), 2, feature_costs=costs)
