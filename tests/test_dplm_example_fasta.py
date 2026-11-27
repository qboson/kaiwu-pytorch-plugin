"""The DPLM examples must ship a usable default FASTA asset."""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "example", "qdiffusion"))

from dplm.utils.io import default_fasta_path, read_fasta_records  # noqa: E402


class TestBundledExampleFasta(unittest.TestCase):
    def test_default_fasta_is_bundled_and_parsable(self):
        path = default_fasta_path()
        self.assertTrue(path.is_file(), f"missing bundled example FASTA: {path}")

        records = read_fasta_records(path)
        self.assertGreaterEqual(len(records), 5)
        self.assertTrue(all(sequence for _, sequence in records))
        self.assertTrue(
            any(50 <= len(sequence) <= 256 for _, sequence in records),
            "at least one record must fit the simple example's 50..256 filter",
        )


if __name__ == "__main__":
    unittest.main()
