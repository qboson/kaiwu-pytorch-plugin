"""Identity-metric contract for the DPLM sequence metrics."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from utils.metrics import sequence_identity  # noqa: E402


def test_identity_is_measured_over_the_shared_prefix():
    """Only the shared prefix contributes to the ratio."""
    assert sequence_identity("ACGT", "ACGA") == 0.75
    assert sequence_identity("ACGTAC", "AC") == 1.0
    assert sequence_identity("AAAA", "TTTT") == 0.0


def test_empty_sequences_have_zero_identity():
    """Either side being empty yields 0.0 instead of dividing by zero."""
    assert sequence_identity("", "ACGT") == 0.0
    assert sequence_identity("ACGT", "") == 0.0
    assert sequence_identity("", "") == 0.0
