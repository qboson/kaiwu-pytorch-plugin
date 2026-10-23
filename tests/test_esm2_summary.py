"""Aggregate contract for the ESM2 distance summaries."""

import os
import sys
import types

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

_esm = types.ModuleType("esm")
_added = "esm" not in sys.modules
if _added:
    sys.modules["esm"] = _esm
from workflows.esm2_eval_helpers import (  # noqa: E402
    PairDistanceRow,
    summarize_distances,
)
if _added:
    sys.modules.pop("esm", None)


def row(index, cosine, l2, reference_length=10, candidate_length=9):
    """Builds one distance row with the metric under test."""
    return PairDistanceRow(
        index=index,
        reference_header=f"r{index}",
        candidate_header=f"c{index}",
        reference_length=reference_length,
        candidate_length=candidate_length,
        cosine_distance=cosine,
        l2_distance=l2,
    )


def test_summary_reports_mean_median_min_and_max():
    """All five statistics come from the row values."""
    rows = [
        row(1, 1.0, 0.5, 10, 8),
        row(2, 2.0, 1.5, 20, 18),
        row(3, 4.0, 5.5, 30, 28),
    ]
    summary = summarize_distances("guided", rows)
    assert summary.label == "guided"
    assert summary.paired_count == 3
    assert summary.mean_reference_length == pytest.approx(20.0)
    assert summary.mean_candidate_length == pytest.approx(18.0)
    assert summary.mean_cosine_distance == pytest.approx(7.0 / 3.0)
    assert summary.median_cosine_distance == pytest.approx(2.0)
    assert summary.min_cosine_distance == pytest.approx(1.0)
    assert summary.max_cosine_distance == pytest.approx(4.0)
    assert summary.mean_l2_distance == pytest.approx(2.5)
    assert summary.median_l2_distance == pytest.approx(1.5)
    assert summary.min_l2_distance == pytest.approx(0.5)
    assert summary.max_l2_distance == pytest.approx(5.5)


def test_even_row_count_uses_the_lower_middle_value():
    """An even number of rows keeps torch's lower median convention."""
    rows = [row(1, 1.0, 1.0), row(2, 2.0, 2.0), row(3, 3.0, 3.0), row(4, 10.0, 10.0)]
    summary = summarize_distances("baseline", rows)
    assert summary.median_cosine_distance == pytest.approx(2.0)
    assert summary.median_l2_distance == pytest.approx(2.0)


def test_without_rows_the_label_is_named_in_the_error():
    """An empty row list is rejected with the evaluation label."""
    with pytest.raises(ValueError, match="baseline"):
        summarize_distances("baseline", [])
