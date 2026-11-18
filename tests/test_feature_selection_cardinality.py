"""FeatureSelectionWrapper must validate cardinality_k at construction."""

from __future__ import annotations

import pytest
import torch

from kaiwu.torch_plugin.maifs import FeatureSelectionWrapper


def _wrapper(**kwargs) -> FeatureSelectionWrapper:
    return FeatureSelectionWrapper(
        model=torch.nn.Linear(4, 1),
        feature_dim=4,
        **kwargs,
    )


def test_negative_cardinality_is_rejected() -> None:
    with pytest.raises(ValueError, match="cardinality_k must be non-negative"):
        _wrapper(cardinality_k=-1)


@pytest.mark.parametrize("value", [1.5, True, "2"])
def test_non_integer_cardinality_is_rejected(value) -> None:
    with pytest.raises(TypeError, match="cardinality_k must be a non-negative integer"):
        _wrapper(cardinality_k=value)


def test_valid_cardinality_is_stored() -> None:
    assert _wrapper(cardinality_k=2).cardinality_k == 2
    assert _wrapper().cardinality_k is None
