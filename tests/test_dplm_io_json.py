"""On-disk JSON contract for the DPLM example reports."""

import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from utils.io import save_json  # noqa: E402


PAYLOAD = {"records": 3, "labels": ["baseline", "guided"]}


def test_save_json_creates_parent_directories_and_round_trips(tmp_path):
    """A nested output path is created before the file is written."""
    target = tmp_path / "nested" / "deeper" / "metrics.json"
    save_json(target, PAYLOAD)
    assert json.loads(target.read_text(encoding="utf-8")) == PAYLOAD


def test_save_json_writes_stable_indented_text(tmp_path):
    """Reports are written with a two-space indent so diffs stay readable."""
    target = tmp_path / "metrics.json"
    save_json(target, PAYLOAD)
    lines = target.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "{"
    assert lines[1] == '  "records": 3,'
    assert lines[-1] == "}"
