"""Delimited-report contract for the DPLM example writers."""

import csv
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from utils.io import write_csv_rows, write_tsv_rows  # noqa: E402


ROWS = [{"index": 1, "label": "baseline"}, {"index": 2, "label": "guided"}]


def test_tsv_writer_uses_first_row_keys_and_tabs(tmp_path):
    """The header comes from the first row and columns are tab separated."""
    target = tmp_path / "summary.tsv"
    write_tsv_rows(target, ROWS)
    assert target.read_text(encoding="utf-8").splitlines() == [
        "index\tlabel",
        "1\tbaseline",
        "2\tguided",
    ]


def test_csv_writer_uses_first_row_keys(tmp_path):
    """The CSV variant keeps the same schema with comma separation."""
    target = tmp_path / "summary.csv"
    write_csv_rows(target, ROWS)
    with target.open(newline="", encoding="utf-8") as handle:
        assert list(csv.reader(handle)) == [
            ["index", "label"],
            ["1", "baseline"],
            ["2", "guided"],
        ]


def test_empty_rows_write_an_empty_file(tmp_path):
    """An empty row list produces no header line at all."""
    tsv = tmp_path / "empty.tsv"
    csv_path = tmp_path / "nested" / "empty.csv"
    write_tsv_rows(tsv, [])
    write_csv_rows(csv_path, [])
    assert tsv.read_text(encoding="utf-8") == ""
    assert csv_path.read_text(encoding="utf-8") == ""
