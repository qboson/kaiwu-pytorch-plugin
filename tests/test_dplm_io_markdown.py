"""Markdown output contract for the DPLM example helpers."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from utils.io import save_markdown  # noqa: E402


def test_markdown_is_joined_with_newlines_and_no_trailing_break(tmp_path):
    """Lines are written verbatim, joined by single newlines."""
    target = tmp_path / "reports" / "note.md"
    save_markdown(target, ["# Title", "", "- item"])
    assert target.read_text(encoding="utf-8") == "# Title\n\n- item"


def test_parent_directories_are_created(tmp_path):
    """A nested report path is created before writing."""
    target = tmp_path / "a" / "b" / "note.md"
    save_markdown(target, ["x"])
    assert target.exists()
