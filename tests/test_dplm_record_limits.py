"""Reject accidental nonpositive FASTA budgets consistently in both workflows."""

import importlib

import pytest

pytest.importorskip("transformers")
pytest.importorskip("omegaconf")
pytest.importorskip("esm")

training = importlib.import_module("example.qdiffusion.dplm.workflows.workflow_helpers")
evaluation = importlib.import_module("example.qdiffusion.dplm.workflows.esm2_eval_helpers")
RECORDS = [("short", "A"), ("first", "AC"), ("second", "DEF"), ("long", "ACDEFG")]


@pytest.mark.parametrize("limit", [0, -1, -10])
@pytest.mark.parametrize("records", [RECORDS, []])
@pytest.mark.parametrize("selector", ["training", "evaluation"])
def test_nonpositive_limits_are_rejected(limit, records, selector):
    """Neither process one unexpected record nor silently slice off tail records."""
    with pytest.raises(ValueError, match="max_records"):
        if selector == "training":
            training.select_records(records, min_length=2, max_length=4, max_records=limit)
        else:
            evaluation.maybe_limit_records(records, limit)


@pytest.mark.parametrize("limit", [1, 2, 10, None])
def test_positive_and_unlimited_selection(limit):
    """Count eligible training records, preserve order, and leave input unchanged."""
    before = list(RECORDS)
    expected = RECORDS[1:3][:limit]
    assert training.select_records(
        RECORDS, min_length=2, max_length=4, max_records=limit,
    ) == expected
    assert evaluation.maybe_limit_records(RECORDS, limit) == RECORDS[:limit]
    assert RECORDS == before
