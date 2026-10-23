"""Artifact schema contract for the ESM2 distance writers."""

import json
import os
import sys
import types

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

_esm = types.ModuleType("esm")
_added = "esm" not in sys.modules
if _added:
    sys.modules["esm"] = _esm
from workflows.esm2_eval_helpers import (  # noqa: E402
    DistanceSummary,
    PairDistanceRow,
    write_rows_csv,
    write_summary_json,
)
if _added:
    sys.modules.pop("esm", None)


ROW = PairDistanceRow(
    index=1,
    reference_header="r1",
    candidate_header="c1",
    reference_length=10,
    candidate_length=9,
    cosine_distance=0.25,
    l2_distance=1.5,
)


def summary():
    """Builds one summary with every field set to a distinct value."""
    return DistanceSummary(
        label="guided",
        paired_count=1,
        mean_reference_length=10.0,
        mean_candidate_length=9.0,
        mean_cosine_distance=0.25,
        median_cosine_distance=0.25,
        min_cosine_distance=0.25,
        max_cosine_distance=0.25,
        mean_l2_distance=1.5,
        median_l2_distance=1.5,
        min_l2_distance=1.5,
        max_l2_distance=1.5,
    )


def test_rows_csv_uses_the_dataclass_field_order(tmp_path):
    """The CSV header is the dataclass field order and one row follows."""
    target = tmp_path / "nested" / "rows.csv"
    write_rows_csv(target, [ROW])
    lines = target.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "index,reference_header,candidate_header,reference_length,"
        "candidate_length,cosine_distance,l2_distance"
    )
    assert lines[1] == "1,r1,c1,10,9,0.25,1.5"


def test_summary_json_contains_every_field(tmp_path):
    """The JSON artifact round-trips all summary fields."""
    target = tmp_path / "summary.json"
    write_summary_json(target, summary())
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload == {
        "label": "guided",
        "paired_count": 1,
        "mean_reference_length": 10.0,
        "mean_candidate_length": 9.0,
        "mean_cosine_distance": 0.25,
        "median_cosine_distance": 0.25,
        "min_cosine_distance": 0.25,
        "max_cosine_distance": 0.25,
        "mean_l2_distance": 1.5,
        "median_l2_distance": 1.5,
        "min_l2_distance": 1.5,
        "max_l2_distance": 1.5,
    }
