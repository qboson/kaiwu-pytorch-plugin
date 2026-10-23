"""Uniqueness-metric contract for the DPLM sequence metrics."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from utils.metrics import uniqueness_ratio  # noqa: E402


def test_ratio_counts_distinct_sequences():
    """Duplicates reduce the ratio proportionally."""
    assert uniqueness_ratio(["A", "A", "B"]) == 2.0 / 3.0
    assert uniqueness_ratio(["A", "B", "C"]) == 1.0
    assert uniqueness_ratio(["A", "A"]) == 0.5


def test_empty_input_reports_zero():
    """An empty generation set has no unique sequences."""
    assert uniqueness_ratio([]) == 0.0
