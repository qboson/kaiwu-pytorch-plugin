"""Report-body contract for the DPLM workflow markdown report."""

import os
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("tqdm")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from workflows.workflow_helpers import write_markdown_report  # noqa: E402
from utils.metrics import QualitySummary  # noqa: E402


def build_quality(jsd, label):
    """Builds one quality summary with distinguishable metric values."""
    return QualitySummary(
        label=label,
        reference_count=4,
        generated_count=4,
        mean_reference_length=10.0,
        mean_generated_length=9.5,
        length_match_ratio=0.75,
        identity_to_reference_mean=0.5,
        amino_acid_jsd=jsd,
        kmer2_jsd=0.2,
        kmer3_jsd=0.3,
        uniqueness_ratio=1.0,
        repeat_ratio_ge4=0.0,
    )


def render(tmp_path, checkpoint=None):
    """Writes one report and returns its lines."""
    target = tmp_path / "report" / "report.md"
    write_markdown_report(
        target,
        config=SimpleNamespace(
            train=SimpleNamespace(num_candidates=6),
            generate=SimpleNamespace(num_candidates=3, steps=8),
        ),
        validation_summary={
            "header": "seq_1 test",
            "raw_length": 12,
            "token_length": 14,
            "generated_length": 11,
            "objective_summary": "logits=(1, 14, 21)",
        },
        train_history=[
            {"epoch": 1, "train_energy_objective_mean": 1.5, "val_energy_objective_mean": 2.5}
        ],
        baseline_quality=build_quality(0.125, "baseline"),
        guided_quality=build_quality(0.375, "guided"),
        comparison_summary={
            "changed_sequences": 4,
            "changed_fraction": 0.5,
            "baseline_identity_mean": 0.25,
            "guided_identity_mean": 0.5,
        },
        selected_count=7,
        train_count=5,
        val_count=1,
        test_count=1,
        best_checkpoint_path=checkpoint,
    )
    return target.read_text(encoding="utf-8").splitlines()


def test_report_contains_every_section(tmp_path):
    """The report keeps its six numbered sections in order."""
    lines = render(tmp_path)
    headings = [line for line in lines if line.startswith("## ")]
    assert headings == [
        "## 1. Data Split",
        "## 2. Structural Validation",
        "## 3. Training History",
        "## 4. Baseline vs Guided Quality",
        "## 5. Baseline vs Guided Difference",
        "## 6. Checkpoint",
    ]


def test_report_embeds_split_counts_and_config(tmp_path):
    """Split counts and both candidate counts are rendered from the inputs."""
    lines = render(tmp_path)
    assert "| selected_records | 7 |" in lines
    assert "| train_records | 5 |" in lines
    assert "| val_records | 1 |" in lines
    assert "| test_records | 1 |" in lines
    assert "| train_num_candidates | 6 |" in lines
    assert "| guided_num_candidates | 3 |" in lines
    assert "| generation_steps | 8 |" in lines


def test_report_formats_history_and_quality_with_fixed_precision(tmp_path):
    """Training and quality rows use five decimals."""
    lines = render(tmp_path)
    assert "| 1 | 1.50000 | 2.50000 |" in lines
    assert "| amino_acid_jsd | 0.12500 | 0.37500 |" in lines
    assert "| changed_sequences | 4 |" in lines
    assert "| changed_fraction | 0.50000 |" in lines


def test_missing_checkpoint_is_reported_as_not_available(tmp_path):
    """Without a best checkpoint the table says ``N/A``."""
    lines = render(tmp_path)
    assert "| best_checkpoint | N/A |" in lines
