"""Result-directory contract for the RBM digits example."""

import os
import sys
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "rbm_digits"))

from rbm_digits import _ensure_result_dir  # noqa: E402


def test_result_directory_is_created_relative_to_the_working_directory(tmp_path, monkeypatch):
    """The helper creates ``results`` under the current working directory."""
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "results").exists()
    _ensure_result_dir()
    assert (tmp_path / "results").is_dir()


def test_calling_twice_is_idempotent(tmp_path, monkeypatch):
    """An existing results directory is accepted instead of raising."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    _ensure_result_dir()
    _ensure_result_dir()
    assert sorted(path.name for path in Path(tmp_path).iterdir()) == ["results"]
