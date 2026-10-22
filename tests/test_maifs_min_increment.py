"""The auto-derived minimum increment must be the smallest positive gap."""

import numpy as np

from kaiwu.torch_plugin.maifs.qubo import PrecisionSplitExplorer


def test_a_user_value_is_returned_unchanged():
    matrix = np.array([[0.0, 0.5], [0.5, 0.0]])

    assert PrecisionSplitExplorer._resolve_min_increment(matrix, 0.1) == 0.1


def test_a_single_distinct_value_uses_one():
    assert PrecisionSplitExplorer._resolve_min_increment(np.zeros((2, 2)), None) == 1


def test_the_smallest_positive_gap_is_used():
    matrix = np.array([[0.0, 0.25], [0.25, 1.0]])

    assert PrecisionSplitExplorer._resolve_min_increment(matrix, None) == 0.25
