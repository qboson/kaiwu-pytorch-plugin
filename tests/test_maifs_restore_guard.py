"""restore_solution must require a completed split plan."""

import numpy as np
import pytest

from kaiwu.torch_plugin.maifs.qubo import PrecisionSplitExplorer


def test_restore_without_a_plan_is_rejected():
    explorer = PrecisionSplitExplorer(target_precision=8, max_bits=10)

    assert explorer.plan is None
    with pytest.raises(ValueError, match="search or fit must be called"):
        explorer.restore_solution(np.zeros(4, dtype=int))
