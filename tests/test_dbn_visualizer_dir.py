"""Result-directory contract for the DBN visualiser."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))

from supervised_dbn_digits import RBMVisualizer  # noqa: E402


def test_constructor_creates_the_result_directory(tmp_path, monkeypatch):
    """The configured directory exists right after construction, even when nested."""
    monkeypatch.chdir(tmp_path)
    RBMVisualizer("nested/results")
    assert (tmp_path / "nested" / "results").is_dir()


def test_existing_directory_is_reused(tmp_path, monkeypatch):
    """A second visualiser with the same directory does not fail."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    RBMVisualizer()
    RBMVisualizer()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["results"]
