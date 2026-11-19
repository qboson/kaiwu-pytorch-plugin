"""ESM2 report distances preserve alignment across bounded tensor batches."""

import pytest
import torch

pytest.importorskip("esm")
pytest.importorskip("omegaconf")
pytest.importorskip("transformers")

from example.qdiffusion.dplm.workflows import esm2_eval_helpers as report


def test_aligned_distances_match_scalar_results_and_include_the_tail(monkeypatch):
    references = [("s1", "AC D"), ("s2", "GG"), ("s3", "TT"), ("skip", "AA")]
    candidates = [("s3", "W"), ("s1", "ACD"), ("s2", "YY")]
    ref_vectors = {
        "s1": torch.tensor([1.0, 0.0]),
        "s2": torch.tensor([0.0, 0.0]),
        "s3": torch.tensor([0.0, 1.0]),
    }
    cand_vectors = {
        "s1": torch.tensor([0.0, 1.0]),
        "s2": torch.tensor([0.0, 1.0]),
        "s3": torch.tensor([0.0, -1.0]),
    }
    batch_sizes = []
    cos = report.F.cosine_similarity

    def record(reference, candidate, dim):
        batch_sizes.append(reference.shape[0])
        return cos(reference, candidate, dim=dim)

    monkeypatch.setattr(report, "_DISTANCE_BATCH_SIZE", 2, raising=False)
    monkeypatch.setattr(report.F, "cosine_similarity", record)
    rows, summary = report.evaluate_candidate_set(
        label="example", reference_records=references, candidate_records=candidates,
        reference_embeddings=ref_vectors, candidate_embeddings=cand_vectors,
        pair_mode="header",
    )

    assert batch_sizes == [2, 1]
    assert [(row.index, row.reference_header, row.candidate_header) for row in rows] == [
        (1, "s1", "s1"), (2, "s2", "s2"), (3, "s3", "s3"),
    ]
    assert [row.reference_length for row in rows] == [3, 2, 2]
    assert [row.cosine_distance for row in rows] == [1.0, 1.0, 2.0]
    assert [row.l2_distance for row in rows] == pytest.approx([2 ** 0.5, 1.0, 2.0])
    assert summary.paired_count == 3
