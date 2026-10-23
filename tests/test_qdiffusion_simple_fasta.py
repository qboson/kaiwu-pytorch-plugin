"""FASTA and decode helper contract for the minimal Q-Diffusion example."""

import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIFF = os.path.join(REPO_ROOT, "example", "qdiffusion")
sys.path[:0] = [QDIFF, os.path.join(QDIFF, "dplm"), os.path.join(QDIFF, "simple")]

import simple_generate_example as example  # noqa: E402


def write_fasta(tmp_path, records, name="input.fasta"):
    """Writes ``(header, sequence)`` pairs in FASTA form."""
    path = tmp_path / name
    lines = []
    for header, sequence in records:
        lines.append(f">{header}")
        lines.append(sequence)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_first_usable_sequence_skips_records_outside_the_range(tmp_path):
    """Too-short and too-long records are skipped in file order."""
    path = write_fasta(
        tmp_path,
        [
            ("short", "AAA"),
            ("long", "A" * 20),
            ("good", "A" * 8),
            ("later", "A" * 9),
        ],
    )
    assert example.first_usable_sequence(path, min_length=5, max_length=10) == (
        "good",
        "A" * 8,
    )


def test_both_length_bounds_are_inclusive(tmp_path):
    """A record exactly at min or max length is usable."""
    lower = write_fasta(tmp_path, [("lower", "A" * 5)], name="lower.fasta")
    upper = write_fasta(tmp_path, [("upper", "A" * 10)], name="upper.fasta")
    assert example.first_usable_sequence(lower, min_length=5, max_length=10)[0] == "lower"
    assert example.first_usable_sequence(upper, min_length=5, max_length=10)[0] == "upper"


def test_without_usable_records_the_scan_raises(tmp_path):
    """An all-too-short file reports the documented error."""
    path = write_fasta(tmp_path, [("short", "AAA")])
    with pytest.raises(ValueError, match="No usable sequence found"):
        example.first_usable_sequence(path, min_length=5, max_length=10)
