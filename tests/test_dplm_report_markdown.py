"""Report generation through exact AST-selected pure production functions.

No scientific module, tokenizer, model or workflow is imported or executed.
"""

from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath
import sys
import tempfile
from types import CodeType, FunctionType, SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "example/qdiffusion/dplm/workflows/workflow_helpers.py"
IO_SOURCE = ROOT / "example/qdiffusion/dplm/utils/io.py"


def load_report_functions(source_path: Path, io_path: Path):
    """Compile complete selected definitions without heavyweight module imports."""
    namespace = {"Path": Path}
    selections = (
        (io_path, {"save_markdown"}),
        (source_path, {"write_markdown_report", "_markdown_table_cell"}),
    )
    for path, selected in selections:
        parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        functions = [
            node for node in parsed.body
            if isinstance(node, ast.FunctionDef) and node.name in selected
        ]
        names = {node.name for node in functions}
        required = selected - {"_markdown_table_cell"}
        if not required.issubset(names):
            raise ValueError(f"Missing production functions in {path}: {required - names}")
        # Future annotations keep the original QualitySummary/config annotations
        # as data; no Torch or model objects are needed by this report function.
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                )
            ] + functions,
            type_ignores=[],
        )
        compiled = compile(ast.fix_missing_locations(module), str(path), "exec")
        function_codes = {
            value.co_name: value for value in compiled.co_consts
            if isinstance(value, CodeType)
        }
        for node in functions:
            if (
                node.decorator_list or node.args.defaults
                or any(default is not None for default in node.args.kw_defaults)
            ):
                raise ValueError("The pure report seam requires undecorated, required arguments")
            code = function_codes[node.name]
            if code.co_freevars:
                raise ValueError("The pure report seam does not reconstruct closure state")
            function = FunctionType(code, namespace, node.name)
            function.__annotations__ = {
                argument.arg: ast.unparse(argument.annotation)
                for argument in node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                if argument.annotation is not None
            }
            if node.returns is not None:
                function.__annotations__["return"] = ast.unparse(node.returns)
            namespace[node.name] = function
    return namespace["write_markdown_report"]


class MarkdownReportTests(unittest.TestCase):
    """Check literal table values and unchanged numeric report formatting."""
    @classmethod
    def setUpClass(cls):
        cls.report = staticmethod(load_report_functions(SOURCE, IO_SOURCE))

    def render(self, *, header="synthetic-record", objective="loss=0.0000", checkpoint=None):
        """Write one synthetic report through the selected production functions."""
        quality = SimpleNamespace(
            amino_acid_jsd=0.1, kmer2_jsd=0.2, kmer3_jsd=0.3,
            length_match_ratio=1.0, identity_to_reference_mean=0.5,
            uniqueness_ratio=1.0, repeat_ratio_ge4=0.0,
        )
        config = SimpleNamespace(
            train=SimpleNamespace(num_candidates=4),
            generate=SimpleNamespace(num_candidates=8, steps=2),
        )
        with tempfile.TemporaryDirectory(prefix="cpqc-markdown-report-") as temporary:
            output = Path(temporary) / "nested" / "REPORT.md"
            self.report(
                output, config=config,
                validation_summary={
                    "header": header, "raw_length": 60, "token_length": 62,
                    "generated_length": 60, "objective_summary": objective,
                },
                train_history=[{
                    "epoch": 1, "train_energy_objective_mean": 0.25,
                    "val_energy_objective_mean": 0.5,
                }],
                baseline_quality=quality, guided_quality=quality,
                comparison_summary={
                    "changed_sequences": 1, "changed_fraction": 0.5,
                    "baseline_identity_mean": 0.4, "guided_identity_mean": 0.5,
                },
                selected_count=12, train_count=8, val_count=2, test_count=2,
                best_checkpoint_path=checkpoint,
            )
            return output.read_text(encoding="utf-8")

    def test_uniprot_header_stays_in_its_value_cell(self):
        """Keep a valid pipe-delimited protein identifier in a single cell."""
        # Synthetic identifier in the documented UniProt header shape.
        report = self.render(header="sp|P00000|SYNTHETIC_HUMAN Test protein")
        self.assertIn(
            "| header | sp\\|P00000\\|SYNTHETIC_HUMAN Test protein |\n", report
        )
        self.assertNotIn("| header | sp|P00000|", report)

    def test_literal_backslash_before_pipe_is_not_lost(self):
        """Preserve a literal backslash as well as the following pipe."""
        report = self.render(header=r"synthetic\|literal")
        self.assertIn("| header | synthetic" + "\\" * 3 + "|literal |\n", report)

    def test_text_cells_cannot_split_a_report_row(self):
        """Escape text metadata consistently and keep each table row on one line."""
        # Caller-supplied report metadata is text, not Markdown row syntax.
        report = self.render(
            objective="synthetic\r\nobjective|detail",
            checkpoint=PurePosixPath("synthetic|checkpoint") / "best.pt",
        )
        self.assertIn("| objective_summary | synthetic objective\\|detail |\n", report)
        self.assertIn("| best_checkpoint | synthetic\\|checkpoint/best.pt |\n", report)

    def test_plain_text_and_numeric_tables_are_unchanged(self):
        """Retain the existing output for simple labels and numeric summaries."""
        report = self.render()
        for row in (
            "| header | synthetic-record |\n",
            "| objective_summary | loss=0.0000 |\n",
            "| raw_length | 60 |\n",
            "| selected_records | 12 |\n",
            "| 1 | 0.25000 | 0.50000 |\n",
            "| amino_acid_jsd | 0.10000 | 0.10000 |\n",
            "| best_checkpoint | N/A |\n",
            "| train_num_candidates | 4 |\n",
        ):
            self.assertIn(row, report)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_dplm_markdown_report.py WORKFLOW_HELPERS IO_SOURCE")
    SOURCE, IO_SOURCE = (Path(value) for value in sys.argv[1:])
    unittest.main(argv=[sys.argv[0]], verbosity=2)
