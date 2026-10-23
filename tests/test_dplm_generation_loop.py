"""Generation-loop contract for the DPLM full example."""

import os
import sys

import pytest
import torch

pytest.importorskip("tqdm")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

import workflows.workflow_helpers as helpers  # noqa: E402


RECORDS = [("h1", "AC GT"), ("h2", "ACGTAC")]


class StubTokenizer:
    """Returns one fixed token row and one spaced decoded sequence."""

    def __call__(self, sequence, **kwargs):
        return {"input_ids": torch.tensor([[1, 3, 2]])}

    def batch_decode(self, tokens, skip_special_tokens=True):
        return ["A C G T"]


class StubGenerator:
    """Minimal generator for the generation loop."""

    def __init__(self):
        self.tokenizer = StubTokenizer()
        self.device = torch.device("cpu")
        self.generate_steps = []

    def objective(self, payload):
        return {"energy_objective": torch.tensor([0.123456])}

    def generate(self, tokens, max_steps):
        self.generate_steps.append(max_steps)
        return tokens


def run(tmp_path, monkeypatch, seeds=None):
    """Runs the generation loop with an optional seed recorder."""
    if seeds is not None:
        def fake_seed(seed):
            seeds.append(seed)

        monkeypatch.setattr(helpers, "seed_torch", fake_seed)
    generator = StubGenerator()
    generated, rows = helpers.run_generation_over_records(
        generator,
        RECORDS,
        max_steps=4,
        seed_base=100,
        output_dir=tmp_path / "out",
        label="guided",
    )
    return generator, generated, rows


def test_rows_report_lengths_energy_and_one_based_indexes(tmp_path, monkeypatch):
    """Every record produces one row with compacted generation length."""
    _generator, generated, rows = run(tmp_path, monkeypatch)
    assert [header for header, _ in generated] == ["h1", "h2"]
    assert [sequence for _, sequence in generated] == ["ACGT", "ACGT"]
    assert [row["index"] for row in rows] == [1, 2]
    assert [row["reference_length"] for row in rows] == [5, 6]
    assert [row["generated_length"] for row in rows] == [4, 4]
    assert [row["energy_objective_mean"] for row in rows] == [0.1235, 0.1235]
    assert [row["label"] for row in rows] == ["guided", "guided"]


def test_generation_budget_and_per_record_seeds_are_applied(tmp_path, monkeypatch):
    """Each record is generated with the requested budget and its own seed."""
    seeds = []
    generator, _generated, _rows = run(tmp_path, monkeypatch, seeds)
    assert generator.generate_steps == [4, 4]
    assert seeds == [101, 102]


def test_artifacts_are_written_with_the_label(tmp_path, monkeypatch):
    """The FASTA and TSV artifacts carry the label in their names and content."""
    run(tmp_path, monkeypatch)
    fasta = tmp_path / "out" / "guided_generated_sequences.fasta"
    tsv = tmp_path / "out" / "guided_generation_summary.tsv"
    assert fasta.read_text(encoding="utf-8").splitlines() == [
        ">seq_1 h1",
        "ACGT",
        ">seq_2 h2",
        "ACGT",
    ]
    lines = tsv.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "index\theader\tlabel\treference_length\tgenerated_length\t"
        "energy_objective_mean"
    )
    assert lines[1] == "1\th1\tguided\t5\t4\t0.1235"
