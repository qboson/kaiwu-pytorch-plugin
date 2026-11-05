"""PrecisionSplitExplorer must validate the options it passes to Kaiwu splitting."""

import numpy as np
import pytest

from kaiwu.torch_plugin.maifs.qubo import PrecisionSplitExplorer

MATRIX = np.array([[0.0, 1.0], [1.0, 0.0]])


def _explorer(**kwargs) -> PrecisionSplitExplorer:
    return PrecisionSplitExplorer(
        target_precision=2,
        max_bits=8,
        max_precision=2,
        start_precision=2,
        **kwargs,
    )


@pytest.mark.parametrize("value", [0, -1, float("nan")])
def test_invalid_min_increment_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="min_increment must be a positive number"):
        _explorer(min_increment=value).search(MATRIX)


@pytest.mark.parametrize("value", [-1.0, float("nan")])
def test_invalid_penalty_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="penalty must be a non-negative number"):
        _explorer(penalty=value).search(MATRIX)


@pytest.mark.parametrize("value", ["yes", 1])
def test_non_boolean_round_to_increment_is_rejected(value) -> None:
    with pytest.raises(TypeError, match="round_to_increment must be a boolean"):
        _explorer(round_to_increment=value).search(MATRIX)


def test_default_options_still_produce_a_plan() -> None:
    plan = _explorer().search(MATRIX)
    assert plan.split_size == 2
