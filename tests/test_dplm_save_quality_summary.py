"""Quality-summary artifact contract for the DPLM metrics."""

import csv
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from utils.metrics import QualitySummary, save_quality_summary  # noqa: E402


def build_summary():
    """Builds one fully populated quality summary."""
    return QualitySummary(
        label="guided",
        reference_count=4,
        generated_count=3,
        mean_reference_length=10.0,
        mean_generated_length=9.5,
        length_match_ratio=0.75,
        identity_to_reference_mean=0.5,
        amino_acid_jsd=0.125,
        kmer2_jsd=0.25,
        kmer3_jsd=0.375,
        uniqueness_ratio=1.0,
        repeat_ratio_ge4=0.0,
    )


def test_both_artifacts_are_written_into_a_nested_directory(tmp_path):
    """The JSON and TSV artifacts are created under the requested directory."""
    output_dir = tmp_path / "nested" / "quality"
    save_quality_summary(output_dir, build_summary())
    payload = json.loads((output_dir / "quality_summary.json").read_text(encoding="utf-8"))
    assert payload["label"] == "guided"
    assert payload["identity_to_reference_mean"] == 0.5
    with (output_dir / "quality_summary.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 1
    assert rows[0]["label"] == "guided"
    assert rows[0]["kmer3_jsd"] == "0.375"


def test_summary_round_trips_every_field(tmp_path):
    """Every dataclass field reaches the JSON payload."""
    save_quality_summary(tmp_path, build_summary())
    payload = json.loads((tmp_path / "quality_summary.json").read_text(encoding="utf-8"))
    assert set(payload) == set(build_summary().__dataclass_fields__)
