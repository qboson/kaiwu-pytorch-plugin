"""The published-result verifier can read a separate experiment directory."""

import json
from pathlib import Path
import subprocess
import sys


def verify_command(root):
    script = Path.cwd() / "example" / "molecular_design" / "scripts" / "verify_results.py"
    return subprocess.run(
        [sys.executable, str(script), "--artifact-root", str(root)],
        capture_output=True, encoding="utf-8", check=False,
    )


def test_missing_artifact_reports_requested_directory(tmp_path):
    root = tmp_path / "实验结果"
    process = verify_command(root)

    assert process.returncode == 2
    assert str(root / "results" / "provenance.json") in process.stderr
    assert "Required experiment artifact missing" in process.stderr


def test_verifier_checks_artifact_hash_relative_to_requested_directory(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "comparison.csv").write_text("corrupted\n", encoding="utf-8")
    (results / "provenance.json").write_text(json.dumps({
        "published_sha256": {"results/comparison.csv": "0" * 64},
    }), encoding="utf-8")

    process = verify_command(tmp_path)

    assert process.returncode != 0
    assert "Published file changed: results/comparison.csv" in process.stderr
