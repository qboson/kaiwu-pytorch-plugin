"""Regression tests for the DPLM FASTA I/O helpers."""

import importlib.util
import sys
from pathlib import Path

_IO_PATH = (
    Path(__file__).resolve().parents[1]
    / "example"
    / "qdiffusion"
    / "dplm"
    / "utils"
    / "io.py"
)

_spec = importlib.util.spec_from_file_location("dplm_io_under_test", _IO_PATH)
io = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = io
_spec.loader.exec_module(io)


def test_write_fasta_records_preserves_headers(tmp_path):
    records = [
        ("sp|P12345|TEST_HUMAN", "ACDEFG"),
        ("sp|Q99999|OTHER_HUMAN", "MNPQRS"),
    ]

    path = tmp_path / "generated.fasta"
    io.write_fasta_records(path, records)

    assert io.read_fasta_records(path) == records


def test_written_fasta_headers_keep_original_identity(tmp_path):
    """Header-mode pairing matches on the exact header string."""
    records = [("ref_1 description", "ACDE")]
    path = tmp_path / "generated.fasta"
    io.write_fasta_records(path, records)

    reread = io.read_fasta_records(path)
    reference_map = {header: sequence for header, sequence in reread}

    assert "ref_1 description" in reference_map
