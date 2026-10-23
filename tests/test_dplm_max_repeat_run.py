"""Repeat-run contract for the DPLM sequence metrics."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from utils.metrics import max_repeat_run  # noqa: E402


def test_longest_run_is_returned():
    """The best run wins even when shorter runs follow it."""
    assert max_repeat_run("AAAB") == 3
    assert max_repeat_run("AAABB") == 3
    assert max_repeat_run("ABBBBA") == 4
    assert max_repeat_run("AABBAA") == 2


def test_trivial_inputs_report_one_or_zero():
    """A single token is a run of one; an empty sequence has no run."""
    assert max_repeat_run("A") == 1
    assert max_repeat_run("") == 0
    assert max_repeat_run("ABAB") == 1
