"""Invalid FASTA structure is reported rather than silently losing records."""
import importlib.util
from pathlib import Path
import pytest


@pytest.fixture
def io():
    path = Path(__file__).resolve().parents[1] / "example/qdiffusion/dplm/utils/io.py"
    spec = importlib.util.spec_from_file_location("fasta_structure_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("text,line", [("AAA\n>valid\nCCC\n", 1),
                                      (">\nAAA\n", 1), (">   \nAAA\n", 1),
                                      (">empty\n>valid\nCCC\n", 1),
                                      (">valid\nAAA\n>empty\n", 3)])
def test_rejects_orphans_empty_headers_and_empty_records(io, tmp_path, text, line):
    path = tmp_path / "bad.fasta"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=f"line {line}"):
        io.read_fasta_records(path)


def test_multiline_records_blank_lines_and_duplicate_headers_are_preserved(io, tmp_path):
    path = tmp_path / "valid.fasta"
    path.write_text("\n>same description\nACD\nEF\n\n>same description\nGH\n", encoding="utf-8")
    assert io.read_fasta_records(path) == [("same description", "ACDEF"), ("same description", "GH")]


def test_empty_file_remains_an_empty_record_set(io, tmp_path):
    path = tmp_path / "empty.fasta"
    path.write_text("\n", encoding="utf-8")
    assert io.read_fasta_records(path) == []
