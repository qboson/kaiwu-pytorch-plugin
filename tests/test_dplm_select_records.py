"""Boundary contract for the DPLM workflow record selection."""

import os
import sys

import pytest

pytest.importorskip("tqdm")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from workflows.workflow_helpers import select_records  # noqa: E402


RECORDS = [("a", "AAA"), ("b", "BBBB"), ("c", "CCCCC"), ("d", "DDDDDD")]


def test_zero_limit_keeps_no_records():
    """A maximum of zero records keeps no records."""
    assert select_records(RECORDS, min_length=1, max_length=10, max_records=0) == []


def test_negative_limit_is_rejected():
    """A negative maximum is not a silent slice bound."""
    with pytest.raises(ValueError, match="max_records"):
        select_records(RECORDS, min_length=1, max_length=10, max_records=-1)


def test_limit_below_available_returns_prefix():
    """A positive maximum truncates the selected records to that many."""
    selected = select_records(RECORDS, min_length=1, max_length=10, max_records=3)
    assert selected == RECORDS[:3]


def test_none_limit_keeps_every_matching_record():
    """``None`` means no limit at all."""
    selected = select_records(RECORDS, min_length=1, max_length=10, max_records=None)
    assert selected == RECORDS
