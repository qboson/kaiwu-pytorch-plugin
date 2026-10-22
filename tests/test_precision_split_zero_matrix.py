"""PrecisionSplitExplorer must diagnose matrices the Kaiwu helpers cannot describe."""

import numpy as np
import pytest

from kaiwu.torch_plugin.maifs.qubo import PrecisionSplitExplorer


def _explorer() -> PrecisionSplitExplorer:
    return PrecisionSplitExplorer(
        target_precision=2,
        max_bits=8,
        max_precision=2,
        start_precision=2,
    )


def test_zero_matrix_raises_a_clear_error() -> None:
    with pytest.raises(RuntimeError, match="cannot describe this Ising matrix"):
        _explorer().search(np.zeros((2, 2)))


def test_non_zero_matrix_still_produces_a_plan() -> None:
    plan = _explorer().search(np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert plan.split_size == 2
    assert plan.source_precision == 2
