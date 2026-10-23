"""Text-compaction contract for decoded FASTA sequences."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from utils.io import normalize_decoded_sequence, normalize_sequence  # noqa: E402


def test_inner_and_outer_spaces_are_removed():
    """Tokenizer spacing between residues is stripped, not only at the ends."""
    assert normalize_sequence("  A C D E \n") == "ACDE"


def test_surrounding_whitespace_is_stripped():
    """Tabs and newlines around the sequence are removed."""
    assert normalize_sequence("\tAC\n") == "AC"


def test_decoded_alias_compacts_the_same_way():
    """The legacy alias shares the compaction behaviour."""
    text = " A C\n"
    assert normalize_decoded_sequence(text) == normalize_sequence(text) == "AC"
