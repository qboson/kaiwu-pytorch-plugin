"""Distance reports use the conventional midpoint median for even samples."""

import importlib
import json
from pathlib import Path
from statistics import median

import pytest
import torch


@pytest.fixture
def helpers(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module(
        "example.qdiffusion.dplm.workflows.esm2_eval_helpers"
    )


@pytest.mark.parametrize("values", [[0.2], [0.9, 0.1, 0.4], [0.2, 0.8],
                                    [0.9, 0.1, 0.6, 0.2], [0.2, 0.2]])
def test_summary_matches_statistical_median(helpers, values):
    rows = [
        helpers.PairDistanceRow(i, f"r{i}", f"c{i}", 3, 4, value, value * 10)
        for i, value in enumerate(values)
    ]
    summary = helpers.summarize_distances("test", rows)
    assert summary.median_cosine_distance == pytest.approx(median(values))
    assert summary.median_l2_distance == pytest.approx(median([v * 10 for v in values]))
    assert summary.mean_cosine_distance == pytest.approx(sum(values) / len(values))
    assert summary.min_l2_distance == pytest.approx(min(values) * 10)
    assert summary.max_l2_distance == pytest.approx(max(values) * 10)
    assert summary.paired_count == len(values)


def test_pair_evaluation_exports_midpoint_medians(helpers, tmp_path):
    references = [("r1", "AAA"), ("r2", "CCC")]
    candidates = [("c1", "AAA"), ("c2", "DDD")]
    rows, summary = helpers.evaluate_candidate_set(
        label="test", reference_records=references, candidate_records=candidates,
        reference_embeddings={"r1": torch.tensor([1., 0.]), "r2": torch.tensor([1., 0.])},
        candidate_embeddings={"c1": torch.tensor([1., 0.]), "c2": torch.tensor([-1., 0.])},
        pair_mode="order",
    )
    assert [row.cosine_distance for row in rows] == [0., 2.]
    assert [row.l2_distance for row in rows] == [0., 2.]
    helpers.write_summary_json(tmp_path / "summary.json", summary)
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved["median_cosine_distance"] == 1.
    assert saved["median_l2_distance"] == 1.
    helpers.write_report(
        tmp_path / "report.md", reference_path=Path("reference.fasta"),
        baseline_path=Path("candidate.fasta"), guided_path=None,
        summaries=[summary], model_name="offline", pair_mode="order", pooling="mean",
    )
    assert "| test | 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |" in (
        tmp_path / "report.md"
    ).read_text(encoding="utf-8")


def test_empty_summary_still_rejected(helpers):
    with pytest.raises(ValueError, match="No aligned rows"):
        helpers.summarize_distances("empty", [])
