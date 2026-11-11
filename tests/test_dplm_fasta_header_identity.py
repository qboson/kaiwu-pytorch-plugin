"""Regression checks for FASTA record identity through the real I/O helpers."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "example/qdiffusion/dplm/utils/io.py"
)
SPEC = importlib.util.spec_from_file_location("dplm_fasta_io_test_module", MODULE_PATH)
FASTA_IO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FASTA_IO)


class FastaHeaderIdentityTests(unittest.TestCase):
    def test_written_headers_and_readback_preserve_record_identity(self):
        records = [("record-alpha", "ACGT"), ("record beta | tag=two", "TTAA")]
        expected_text = ">record-alpha\nACGT\n>record beta | tag=two\nTTAA\n"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "records.fasta"
            FASTA_IO.write_fasta_records(path, records)
            actual = (
                path.read_text(encoding="utf-8"),
                FASTA_IO.read_fasta_records(path),
            )
            self.assertEqual(actual, (expected_text, records))

    def test_repeated_roundtrips_do_not_accumulate_or_strip_prefixes(self):
        records = [("seq_7 original label", "AC"), ("second label", "GT")]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.fasta"
            current = list(records)
            for cycle in range(2):
                with self.subTest(cycle=cycle):
                    FASTA_IO.write_fasta_records(path, current)
                    current = FASTA_IO.read_fasta_records(path)
                    self.assertEqual(current, records)

    def test_empty_records_create_parent_and_empty_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "new" / "nested" / "empty.fasta"
            FASTA_IO.write_fasta_records(path, [])
            self.assertTrue(path.is_file())
            self.assertEqual(path.read_bytes(), b"")
            self.assertEqual(FASTA_IO.read_fasta_records(path), [])


if __name__ == "__main__":
    unittest.main()
