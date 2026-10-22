"""update_mask must store the projected mask it returns."""

import numpy as np
import torch
from torch import nn

from kaiwu.torch_plugin.maifs import FeatureSelectionWrapper


def test_update_mask_stores_the_returned_mask() -> None:
    torch.manual_seed(0)
    selector = FeatureSelectionWrapper(
        nn.Linear(4, 1),
        feature_dim=4,
        min_selected_features=1,
        max_selected_features=3,
    )
    inputs = torch.randn(8, 4)
    targets = inputs[:, :1] * 2.0
    loader = [(inputs, targets)]

    selected = selector.update_mask(loader, nn.MSELoss())

    assert selected.shape == (4,)
    assert np.array_equal(selected, selector.get_support().astype(int))
    assert np.array_equal(selected, selector.mask.detach().numpy().astype(int))
    assert int(selected.sum()) <= 3
