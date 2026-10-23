"""Residue-count contract for the DPLM sequence metrics."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from utils.metrics import token_distribution  # noqa: E402


def test_counts_are_accumulated_across_sequences():
    """Frequencies are summed over every sequence in the set."""
    counts = token_distribution(["AAB", "A"])
    assert counts == {"A": 3, "B": 1}


def test_empty_sequence_set_has_no_counts():
    """No sequences means an empty mapping."""
    assert token_distribution([]) == {}
    assert token_distribution(["", ""]) == {}
