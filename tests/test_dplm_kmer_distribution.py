"""K-mer-count contract for the DPLM sequence metrics."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from utils.metrics import kmer_distribution  # noqa: E402


def test_all_sliding_windows_are_counted():
    """Every length-k window contributes once, including repeated windows."""
    counts = kmer_distribution(["AAAB"], 2)
    assert counts == {"AA": 2, "AB": 1}


def test_short_sequences_are_skipped():
    """Sequences shorter than ``k`` contribute nothing."""
    counts = kmer_distribution(["AC", "A"], 3)
    assert counts == {}
    assert kmer_distribution(["ACG"], 3) == {"ACG": 1}


def test_counts_accumulate_across_sequences():
    """The same window in different sequences is summed."""
    counts = kmer_distribution(["AAA", "AAA"], 2)
    assert counts == {"AA": 4}
