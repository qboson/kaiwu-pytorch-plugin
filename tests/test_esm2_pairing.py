"""Record-pairing contract for the ESM2 distance evaluation helpers."""

import os
import sys
import types

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

# The helper module imports ``esm`` at import time; pairing never touches it.
_esm = types.ModuleType("esm")
_added = "esm" not in sys.modules
if _added:
    sys.modules["esm"] = _esm
from workflows.esm2_eval_helpers import (  # noqa: E402
    pair_records_by_header,
    pair_records_by_order,
)
if _added:
    sys.modules.pop("esm", None)


REFERENCE = [("h1", "AAA"), ("h2", "BBB"), ("h3", "CCC")]


def test_order_mode_pairs_positionally_and_truncates_to_the_shorter_side():
    """Order mode pairs the first N records of both sides."""
    candidate = [("c1", "XXX"), ("c2", "YYY")]
    pairs = pair_records_by_order(REFERENCE, candidate)
    assert pairs == [
        (("h1", "AAA"), ("c1", "XXX")),
        (("h2", "BBB"), ("c2", "YYY")),
    ]
    longer = pair_records_by_order(REFERENCE, candidate + [("c3", "ZZZ")])
    assert len(longer) == 3
    assert longer[-1] == (("h3", "CCC"), ("c3", "ZZZ"))


def test_header_mode_keeps_reference_order_and_drops_unmatched_records():
    """Header mode follows the reference order regardless of the candidate order."""
    candidate = [("h3", "yyy"), ("h4", "zzz"), ("h1", "xxx")]
    pairs = pair_records_by_header(REFERENCE, candidate)
    assert pairs == [
        (("h1", "AAA"), ("h1", "xxx")),
        (("h3", "CCC"), ("h3", "yyy")),
    ]


def test_header_mode_keeps_the_last_duplicate_candidate():
    """A repeated candidate header resolves to the later record."""
    candidate = [("h2", "first"), ("h2", "second")]
    pairs = pair_records_by_header(REFERENCE, candidate)
    assert pairs == [(("h2", "BBB"), ("h2", "second"))]


def test_header_mode_without_overlap_returns_no_pairs():
    """No shared headers means no pairs rather than an error at this layer."""
    assert pair_records_by_header(REFERENCE, [("other", "AAA")]) == []
