"""Log-directory contract for the BM generation saver."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "bm_generation"))

from saver import Saver  # noqa: E402


def test_constructor_creates_the_log_directory(tmp_path):
    """The configured log directory exists right after construction."""
    log_dir = tmp_path / "nested" / "logs"
    Saver(log_path=str(log_dir))
    assert log_dir.is_dir()


def test_output_loss_appends_one_line_per_call(tmp_path):
    """Every call appends a single formatted line to ``loss.txt``."""
    log_dir = tmp_path / "logs"
    saver = Saver(log_path=str(log_dir))
    saver.output_loss(1, 0.5, 2, "f")
    saver.output_loss(2, 0.25, 4, "g")
    content = (log_dir / "loss.txt").read_text(encoding="utf-8")
    assert content.splitlines() == [
        "step:1, kl_div:0.5, ncl:2, cost:f",
        "step:2, kl_div:0.25, ncl:4, cost:g",
    ]


def test_existing_log_directory_is_reused(tmp_path):
    """An already existing directory is accepted and keeps old content."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "loss.txt").write_text("old\n", encoding="utf-8")
    saver = Saver(log_path=str(log_dir))
    saver.output_loss(3, 1.0, 6, "h")
    lines = (log_dir / "loss.txt").read_text(encoding="utf-8").splitlines()
    assert lines == ["old", "step:3, kl_div:1.0, ncl:6, cost:h"]
