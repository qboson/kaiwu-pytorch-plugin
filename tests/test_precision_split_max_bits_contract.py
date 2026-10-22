"""Pin the max_bits guard of PrecisionSplitExplorer.search."""

import numpy as np
import pytest

from kaiwu.torch_plugin.maifs.qubo import PrecisionSplitExplorer


def test_matrix_larger_than_max_bits_is_rejected() -> None:
    explorer = PrecisionSplitExplorer(
        target_precision=2,
        max_bits=2,
        max_precision=2,
        start_precision=2,
    )
    with pytest.raises(ValueError, match="cannot be larger than max_bits"):
        explorer.search(np.ones((3, 3)))


def test_matrix_within_max_bits_still_searches() -> None:
    explorer = PrecisionSplitExplorer(
        target_precision=2,
        max_bits=2,
        max_precision=2,
        start_precision=2,
    )
    plan = explorer.search(np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert plan.split_size <= 2
