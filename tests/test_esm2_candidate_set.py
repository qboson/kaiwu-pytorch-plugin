"""Distance-math contract for one ESM2 candidate set."""

import math
import os
import sys
import types

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

_esm = types.ModuleType("esm")
_added = "esm" not in sys.modules
if _added:
    sys.modules["esm"] = _esm
from workflows.esm2_eval_helpers import evaluate_candidate_set  # noqa: E402
if _added:
    sys.modules.pop("esm", None)


def evaluate(**overrides):
    """Calls the evaluator with one orthogonal pair by default."""
    arguments = {
        "label": "guided",
        "reference_records": [("r1", "A C")],
        "candidate_records": [("c1", "AC")],
        "reference_embeddings": {"r1": torch.tensor([1.0, 0.0])},
        "candidate_embeddings": {"c1": torch.tensor([0.0, 1.0])},
        "pair_mode": "order",
    }
    arguments.update(overrides)
    return evaluate_candidate_set(**arguments)


def test_rows_use_the_geometry_and_normalized_lengths():
    """Cosine distance, L2 distance, lengths and indexes come from the pair."""
    rows, summary = evaluate()
    assert len(rows) == 1
    row = rows[0]
    assert row.index == 1
    assert row.reference_header == "r1"
    assert row.candidate_header == "c1"
    assert row.reference_length == 2
    assert row.candidate_length == 2
    assert row.cosine_distance == pytest.approx(1.0)
    assert row.l2_distance == pytest.approx(math.sqrt(2.0))
    assert summary.paired_count == 1
    assert summary.mean_cosine_distance == pytest.approx(1.0)


def test_header_mode_selects_embeddings_by_header():
    """Header pairing uses the candidate header for the embedding lookup."""
    rows, _summary = evaluate(
        reference_records=[("a", "A"), ("b", "B")],
        candidate_records=[("b", "B"), ("a", "A")],
        reference_embeddings={"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 1.0])},
        candidate_embeddings={"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([1.0, 0.0])},
        pair_mode="header",
    )
    assert [row.reference_header for row in rows] == ["a", "b"]
    assert rows[0].cosine_distance == pytest.approx(0.0)
    assert rows[1].cosine_distance == pytest.approx(1.0)


def test_unknown_pair_mode_is_rejected():
    """An unsupported mode fails before any embedding lookup."""
    with pytest.raises(ValueError, match="Unsupported pair mode"):
        evaluate(pair_mode="random")


def test_no_aligned_pairs_is_rejected_with_the_label():
    """Empty overlap reports the label and the checked inputs."""
    with pytest.raises(ValueError, match="guided"):
        evaluate(
            reference_records=[("r1", "A")],
            candidate_records=[("other", "A")],
            pair_mode="header",
            candidate_embeddings={"other": torch.tensor([1.0, 0.0])},
            reference_embeddings={"r1": torch.tensor([1.0, 0.0])},
        )
