"""Markdown report contract for the ESM2 distance evaluation."""

import os
import sys
import types

import pytest

pytest.importorskip("tqdm")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

_esm = types.ModuleType("esm")
_added = "esm" not in sys.modules
if _added:
    sys.modules["esm"] = _esm
from workflows.esm2_eval_helpers import (  # noqa: E402
    DistanceSummary,
    write_report,
)
if _added:
    sys.modules.pop("esm", None)


def summary(label, cosine, l2):
    """Builds one summary whose cosine/L2 statistics all share one value."""
    return DistanceSummary(
        label=label,
        paired_count=200,
        mean_reference_length=100.0,
        mean_candidate_length=99.0,
        mean_cosine_distance=cosine,
        median_cosine_distance=cosine,
        min_cosine_distance=cosine,
        max_cosine_distance=cosine,
        mean_l2_distance=l2,
        median_l2_distance=l2,
        min_l2_distance=l2,
        max_l2_distance=l2,
    )


def render(tmp_path, summaries, baseline_path, guided_path):
    """Writes one report and returns its lines."""
    target = tmp_path / "reports" / "esm2.md"
    write_report(
        target,
        reference_path=tmp_path / "ref.fasta",
        baseline_path=baseline_path,
        guided_path=guided_path,
        summaries=summaries,
        model_name="esm2_t33_650M_UR50D",
        pair_mode="order",
        pooling="mean",
    )
    return target.read_text(encoding="utf-8").splitlines()


def test_report_lists_paths_settings_and_six_decimal_rows(tmp_path):
    """With two summaries the table and the delta section are both rendered."""
    lines = render(
        tmp_path,
        [summary("baseline", 0.5, 4.0), summary("guided", 0.25, 2.5)],
        tmp_path / "baseline.fasta",
        tmp_path / "guided.fasta",
    )
    assert f"- reference: `{tmp_path / 'ref.fasta'}`" in lines
    assert f"- baseline: `{tmp_path / 'baseline.fasta'}`" in lines
    assert f"- guided: `{tmp_path / 'guided.fasta'}`" in lines
    assert "- esm2 model: `esm2_t33_650M_UR50D`" in lines
    assert "- pair mode: `order`" in lines
    assert "- pooling: `mean`" in lines
    assert "| baseline | 200 | 0.500000 | 0.500000 | 4.000000 | 4.000000 |" in lines
    assert "| guided | 200 | 0.250000 | 0.250000 | 2.500000 | 2.500000 |" in lines
    assert "## Delta" in lines
    assert "- guided minus baseline mean cosine distance: -0.250000" in lines
    assert "- guided minus baseline mean l2 distance: -1.500000" in lines


def test_report_marks_missing_inputs_and_skips_the_delta(tmp_path):
    """A single summary reports the absent inputs and has no delta section."""
    lines = render(tmp_path, [summary("baseline", 0.5, 4.0)], None, None)
    assert "- baseline: not provided" in lines
    assert "- guided: not provided" in lines
    assert "## Delta" not in lines
