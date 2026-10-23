"""Chunking contract for the ESM2 embedding batches."""

import os
import sys
import types

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

_esm = types.ModuleType("esm")
_added = "esm" not in sys.modules
if _added:
    sys.modules["esm"] = _esm
from workflows.esm2_eval_helpers import chunked  # noqa: E402
if _added:
    sys.modules.pop("esm", None)


RECORDS = [(str(index), "A" * index) for index in range(1, 6)]


def test_chunks_cover_every_record_in_order():
    """Concatenating the chunks reproduces the input order."""
    chunks = list(chunked(RECORDS, 2))
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert [record for chunk in chunks for record in chunk] == RECORDS


def test_exact_multiple_has_no_trailing_empty_chunk():
    """A batch size that divides the input yields only full chunks."""
    assert [len(chunk) for chunk in chunked(RECORDS[:4], 2)] == [2, 2]


def test_empty_input_yields_no_chunks():
    """An empty record list produces no batches."""
    assert list(chunked([], 3)) == []


def test_batch_larger_than_input_yields_one_chunk():
    """A single oversized batch still contains every record."""
    chunks = list(chunked(RECORDS, 100))
    assert len(chunks) == 1
    assert chunks[0] == RECORDS
