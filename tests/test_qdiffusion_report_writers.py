"""Behavioral coverage for the DPLM example's I/O and sequence helpers."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


_IO_PATH = (
    Path(__file__).resolve().parents[1]
    / "example" / "qdiffusion" / "dplm" / "utils" / "io.py"
)
# Execute the whole stdlib module without importing model-backed package exports.
_SPEC = importlib.util.spec_from_file_location("_qdiffusion_writer_test_io", _IO_PATH)
_IO = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_IO)


class TestReportWriters(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / ".tmp"
        temporary_root.mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(
            prefix="qdiffusion-reports-", dir=str(temporary_root)
        )
        self.addCleanup(directory.cleanup)
        self.output_root = Path(directory.name)

    def test_json_creates_parents_and_uses_two_space_indentation(self):
        path = self.output_root / "nested" / "json" / "report.json"
        payload = {"report": {"counts": [1, 2]}}
        self.assertFalse(path.parent.exists())

        _IO.save_json(path, payload)

        text = path.read_text(encoding="utf-8")
        self.assertEqual(json.loads(text), payload)
        self.assertEqual(
            text.splitlines(),
            [
                "{",
                '  "report": {',
                '    "counts": [',
                "      1,",
                "      2",
                "    ]",
                "  }",
                "}",
            ],
        )

    def test_csv_preserves_first_row_columns_and_quoted_values(self):
        path = self.output_root / "nested" / "csv" / "report.csv"
        rows = [
            {"label": 'first, "quoted"', "score": 7, "note": "line 1\nline 2"},
            {"note": "plain", "score": 0, "label": "second"},
        ]
        self.assertFalse(path.parent.exists())

        _IO.write_csv_rows(path, rows)

        self.assertEqual(path.read_text(encoding="utf-8").splitlines()[0], "label,score,note")
        with path.open(encoding="utf-8", newline="") as handle:
            self.assertEqual(
                list(csv.reader(handle)),
                [
                    ["label", "score", "note"],
                    ['first, "quoted"', "7", "line 1\nline 2"],
                    ["second", "0", "plain"],
                ],
            )

    def test_tsv_preserves_first_row_columns_and_quoted_values(self):
        path = self.output_root / "nested" / "tsv" / "report.tsv"
        rows = [
            {"label": "first\tfield", "score": 7, "note": 'say "hello"\nagain'},
            {"note": "plain", "score": 0, "label": "second"},
        ]
        self.assertFalse(path.parent.exists())

        _IO.write_tsv_rows(path, rows)

        self.assertEqual(path.read_text(encoding="utf-8").splitlines()[0], "label\tscore\tnote")
        with path.open(encoding="utf-8", newline="") as handle:
            self.assertEqual(
                list(csv.reader(handle, delimiter="\t")),
                [
                    ["label", "score", "note"],
                    ["first\tfield", "7", 'say "hello"\nagain'],
                    ["second", "0", "plain"],
                ],
            )

    def test_empty_delimited_rows_create_empty_files(self):
        for extension, writer in (
            ("csv", _IO.write_csv_rows),
            ("tsv", _IO.write_tsv_rows),
        ):
            with self.subTest(extension=extension):
                path = self.output_root / "nested" / extension / ("empty." + extension)
                self.assertFalse(path.parent.exists())

                writer(path, [])

                self.assertEqual(path.read_bytes(), b"")

    def test_markdown_joins_lines_without_a_trailing_newline(self):
        path = self.output_root / "nested" / "markdown" / "report.md"
        self.assertFalse(path.parent.exists())

        _IO.save_markdown(path, ["# Summary", "", "Result: OK"])

        self.assertEqual(path.read_text(encoding="utf-8"), "# Summary\n\nResult: OK")

    def test_empty_markdown_lines_create_an_empty_file(self):
        path = self.output_root / "nested" / "markdown" / "empty.md"
        self.assertFalse(path.parent.exists())

        _IO.save_markdown(path, [])

        self.assertEqual(path.read_bytes(), b"")


class TestSequenceNormalization(unittest.TestCase):
    def test_helpers_remove_internal_literal_spaces(self):
        for helper in (_IO.normalize_sequence, _IO.normalize_decoded_sequence):
            for text, expected in (("A C D E", "ACDE"), ("A  C   D", "ACD")):
                with self.subTest(helper=helper.__name__, text=text):
                    self.assertEqual(helper(text), expected)

    def test_helpers_trim_endpoint_whitespace(self):
        for helper in (_IO.normalize_sequence, _IO.normalize_decoded_sequence):
            for text, expected in (("\tAC\n", "AC"), ("\r\nACDE\t", "ACDE")):
                with self.subTest(helper=helper.__name__, text=text):
                    self.assertEqual(helper(text), expected)

    def test_alias_and_normalizer_match_explicit_combined_outputs(self):
        for text, expected in (("  A C D E \n", "ACDE"), ("ACDE", "ACDE")):
            with self.subTest(text=text):
                self.assertEqual(_IO.normalize_sequence(text), expected)
                self.assertEqual(_IO.normalize_decoded_sequence(text), expected)


if __name__ == "__main__":
    unittest.main()
