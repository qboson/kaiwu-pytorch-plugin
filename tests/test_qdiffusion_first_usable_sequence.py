"""Behavioral tests for the simple example's real FASTA scanner."""
import importlib
from pathlib import Path
import sys
import tempfile
import unittest


# Match the script's sibling-import context and use its genuine bootstrap.
_SIMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "qdiffusion" / "simple"


def _load_example_module():
    module_names = ("simple_generate_example", "_example_bootstrap")
    missing = object()
    previous_modules = {
        name: sys.modules.get(name, missing) for name in module_names
    }
    previous_path = list(sys.path)
    try:
        for name in module_names:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(_SIMPLE_DIR))
        return importlib.import_module("simple_generate_example")
    finally:
        sys.path[:] = previous_path
        for name, previous in previous_modules.items():
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


_EXAMPLE_MODULE = _load_example_module()
first_usable_sequence = _EXAMPLE_MODULE.first_usable_sequence


class TestFirstUsableSequence(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / ".tmp"
        temporary_root.mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(
            prefix="qdiffusion-fasta-", dir=str(temporary_root)
        )
        self.addCleanup(directory.cleanup)
        self.fasta_path = Path(directory.name) / "input.fasta"

    def write_fasta(self, text):
        self.fasta_path.write_text(text, encoding="utf-8")
        return self.fasta_path

    def test_returns_first_eligible_record_after_rejecting_both_sides(self):
        path = self.write_fasta(
            ">short\nAA\n"
            ">long\nACDEFGH\n"
            ">first eligible\nCDEF\n"
            ">later eligible\nGHIKL\n"
        )
        self.assertEqual(
            first_usable_sequence(path, min_length=3, max_length=6),
            ("first eligible", "CDEF"),
        )

    def test_inclusive_boundaries_at_first_middle_and_final_positions(self):
        for sequence in ("ACD", "ACDEFG"):
            for position in (0, 1, 2):
                with self.subTest(sequence=sequence, position=position):
                    records = [("short", "AA"), ("long", "ACDEFGH")]
                    records.insert(position, ("boundary", sequence))
                    path = self.write_fasta(
                        "".join(
                            ">{}\n{}\n".format(header, body)
                            for header, body in records
                        )
                    )
                    self.assertEqual(
                        first_usable_sequence(path, min_length=3, max_length=6),
                        ("boundary", sequence),
                    )

    def test_final_record_without_trailing_newline(self):
        path = self.write_fasta(
            ">short\nAA\n>last eligible\nDEFGH"
        )
        self.assertEqual(
            first_usable_sequence(path, min_length=4, max_length=6),
            ("last eligible", "DEFGH"),
        )

    def test_single_record_at_eof(self):
        path = self.write_fasta(">only record\nACDE")
        self.assertEqual(
            first_usable_sequence(path, min_length=3, max_length=6),
            ("only record", "ACDE"),
        )

    def test_multiline_sequence_with_blank_lines(self):
        path = self.write_fasta(
            "\n\n>short\nA\n\n"
            ">joined record\nAC\n\nDE\n \t\nFG\n\n"
            ">later eligible\nHIKLM\n"
        )
        self.assertEqual(
            first_usable_sequence(path, min_length=5, max_length=7),
            ("joined record", "ACDEFG"),
        )

    def test_no_match_raises_the_documented_value_error(self):
        cases = (
            "",
            ">short\nAA\n",
            ">long\nACDEFGH\n",
            ">short\nAA\n>long\nACDEFGH",
        )
        for contents in cases:
            with self.subTest(contents=contents):
                path = self.write_fasta(contents)
                with self.assertRaisesRegex(
                    ValueError, r"^No usable sequence found in FASTA\.$"
                ):
                    first_usable_sequence(path, min_length=3, max_length=6)


if __name__ == "__main__":
    unittest.main()
