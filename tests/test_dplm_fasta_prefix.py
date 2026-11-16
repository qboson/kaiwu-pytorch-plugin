"""Read DPLM protein sequences from plain and gzip FASTA files."""

import gzip
from pathlib import Path
import runpy

import pytest


@pytest.fixture
def read_fasta():
    # This I/O helper can be exercised without importing pretrained model dependencies.
    source = Path(__file__).resolve().parents[1] / "example" / "qdiffusion" / "dplm" / "utils" / "io.py"
    return runpy.run_path(str(source))["read_fasta_records"]


@pytest.fixture(params=["fasta", "fasta.gz"])
def write_fasta(request, tmp_path):
    path = tmp_path / f"参考序列.{request.param}"

    def write(content):
        if path.suffix == ".gz":
            with gzip.open(path, "wb") as stream:
                stream.write(content)
        else:
            path.write_bytes(content)
        return path

    return write


def test_reads_complete_multiline_records(read_fasta, write_fasta):
    path = write_fasta(">参考序列 A\nAC\nDE\n\n>B\nF\n".encode("utf-8"))

    assert read_fasta(path) == [("参考序列 A", "ACDE"), ("B", "F")]
    assert read_fasta(path, max_records=1) == [("参考序列 A", "ACDE")]


def test_prefix_does_not_decode_the_unused_tail(read_fasta, write_fasta):
    path = write_fasta(b">included\nAC\n>unused\n" + b"A" * 40000 + b"\xff")

    assert read_fasta(path, max_records=1) == [("included", "AC")]
    assert read_fasta(path, max_records=0) == []
    with pytest.raises(UnicodeDecodeError):
        read_fasta(path)


def test_negative_prefix_limit_is_invalid(read_fasta, write_fasta):
    path = write_fasta(b">included\nAC\n")
    with pytest.raises(ValueError, match="max_records must be non-negative"):
        read_fasta(path, max_records=-1)
