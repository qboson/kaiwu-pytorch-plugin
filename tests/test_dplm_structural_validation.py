"""Structural-validation contract for the DPLM full example."""

import os
import sys

import pytest
import torch

pytest.importorskip("tqdm")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from workflows.workflow_helpers import run_structural_validation  # noqa: E402


TARGET_TEXT = " A C G T " * 20
GENERATED_TEXT = "A C G T " * 30


class StubTokenizer:
    """Returns shape-stable tokens and two canned decoded texts."""

    def __init__(self):
        self.decoded = 0

    def __call__(self, sequence, **kwargs):
        return {"input_ids": torch.tensor([[1] + [3] * 10 + [2]])}

    def batch_decode(self, tokens, skip_special_tokens=True):
        text = [TARGET_TEXT, GENERATED_TEXT][self.decoded]
        self.decoded += 1
        return [text]


class StubGenerator:
    """Minimal generator implementing the structural-validation surface."""

    def __init__(self):
        self.tokenizer = StubTokenizer()
        self.device = torch.device("cpu")
        self.generate_steps = []

    def objective(self, payload):
        targets = payload["targets"]
        return {
            "logits": torch.zeros(1, targets.shape[1], 5),
            "targets": targets,
            "loss_mask": torch.tensor([[True, True, True, False]]),
            "weight": torch.tensor([0.5]),
            "energy_objective": torch.tensor([0.25]),
        }

    def generate(self, tokens, max_steps):
        self.generate_steps.append(max_steps)
        return tokens


def test_lengths_come_from_the_record_and_the_encoded_tokens():
    """Raw length, token width and compacted generation length are reported."""
    summary = run_structural_validation(
        StubGenerator(), ("seq_1 demo", "ACGTACGTAC"), max_length=None, steps=5
    )
    assert summary["header"] == "seq_1 demo"
    assert summary["raw_length"] == 10
    assert summary["token_length"] == 12
    assert summary["generated_length"] == len(GENERATED_TEXT.replace(" ", ""))


def test_both_prefixes_are_truncated_to_eighty_characters():
    """The decoded target and generated text are both clipped for the report."""
    summary = run_structural_validation(
        StubGenerator(), ("seq_1 demo", "ACGTACGTAC"), max_length=None, steps=5
    )
    assert summary["decoded_target_prefix"] == TARGET_TEXT[:80]
    assert summary["generated_prefix"] == GENERATED_TEXT.replace(" ", "")[:80]
    assert len(summary["generated_prefix"]) == 80


def test_objective_summary_and_generation_budget_are_wired():
    """The objective summary is rendered and ``steps`` reaches ``generate``."""
    generator = StubGenerator()
    summary = run_structural_validation(
        generator, ("seq_1 demo", "ACGTACGTAC"), max_length=None, steps=7
    )
    assert generator.generate_steps == [7]
    assert "masked_positions=3" in summary["objective_summary"]
    assert "energy_objective_mean=0.2500" in summary["objective_summary"]
