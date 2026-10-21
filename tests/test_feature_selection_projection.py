"""Pin the feature-count projection performed by update_mask."""

import torch
from torch import nn

from kaiwu.torch_plugin.maifs import FeatureSelectionWrapper


def _loader(signal: bool, rows: int = 32, features: int = 6):
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(rows, features, generator=generator)
    targets = inputs[:, :1] * 3.0 if signal else torch.zeros(rows, 1)
    return [(inputs, targets)]


def _selector(**kwargs) -> FeatureSelectionWrapper:
    model = nn.Linear(6, 1, bias=False)
    return FeatureSelectionWrapper(model, feature_dim=6, **kwargs)


def test_projection_respects_the_explicit_bounds() -> None:
    selector = _selector(min_selected_features=2, max_selected_features=4)
    mask = selector.update_mask(_loader(signal=True), nn.MSELoss())

    assert mask.shape == (6,)
    assert set(torch.as_tensor(mask).tolist()).issubset({0, 1})
    assert 2 <= int(mask.sum()) <= 4
    assert selector.num_selected() == int(mask.sum())
    assert selector.get_support().dtype == bool


def test_projection_enforces_the_upper_bound() -> None:
    selector = _selector(max_selected_features=2)
    mask = selector.update_mask(_loader(signal=True), nn.MSELoss())
    assert int(mask.sum()) <= 2


def test_zero_candidate_still_selects_at_least_one_feature() -> None:
    selector = _selector(min_selected_features=1, max_selected_features=3)
    mask = selector.update_mask(_loader(signal=False), nn.MSELoss())
    assert int(mask.sum()) >= 1
