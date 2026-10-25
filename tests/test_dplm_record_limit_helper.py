"""Contract test for ``dplm.workflows.esm2_eval_helpers.maybe_limit_records``."""

from __future__ import annotations

import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QDIFFUSION = REPO / "example" / "qdiffusion"


def _load_helper():
    esm_stub = types.ModuleType("esm")
    previous = sys.modules.get("esm")
    sys.modules["esm"] = esm_stub
    sys.path.insert(0, str(QDIFFUSION))
    try:
        from dplm.workflows.esm2_eval_helpers import maybe_limit_records
    finally:
        sys.path.remove(str(QDIFFUSION))
        if previous is None:
            sys.modules.pop("esm", None)
        else:
            sys.modules["esm"] = previous
    return maybe_limit_records


RECORDS = [("h1", "AAA"), ("h2", "CCC"), ("h3", "GGG")]


def test_none_keeps_the_input_object_untouched():
    maybe_limit_records = _load_helper()

    result = maybe_limit_records(RECORDS, None)

    assert result is RECORDS


def test_zero_returns_an_empty_prefix():
    maybe_limit_records = _load_helper()

    assert maybe_limit_records(RECORDS, 0) == []


def test_positive_limits_keep_the_first_records():
    maybe_limit_records = _load_helper()

    assert maybe_limit_records(RECORDS, 2) == [("h1", "AAA"), ("h2", "CCC")]


def test_a_limit_above_the_record_count_keeps_everything():
    maybe_limit_records = _load_helper()

    assert maybe_limit_records(RECORDS, 99) == RECORDS
    assert RECORDS == [("h1", "AAA"), ("h2", "CCC"), ("h3", "GGG")]
