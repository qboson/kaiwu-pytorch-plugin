"""Formatting contract for the DPLM objective summary line."""

import os
import sys

import pytest
import torch

pytest.importorskip("tqdm")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from workflows.workflow_helpers import summarize_objective  # noqa: E402


def make_outputs():
    """Builds the minimal objective dictionary the summary formats."""
    return {
        "logits": torch.zeros(2, 3, 4),
        "targets": torch.zeros(2, 3, dtype=torch.long),
        "loss_mask": torch.tensor([[True, False, True], [True, True, False]]),
        "weight": torch.tensor([0.0, 1.0]),
        "energy_objective": torch.tensor([0.5, 0.0]),
        "positive_energy_mean": torch.tensor(0.75),
        "negative_energy_mean": torch.tensor(1.25),
    }


def test_summary_reports_shapes_counts_and_four_decimals():
    """Every field is rendered from the tensors with fixed precision."""
    summary = summarize_objective(make_outputs())
    assert summary == (
        "logits=(2, 3, 4), targets=(2, 3), masked_positions=4, "
        "weight_mean=0.5000, energy_objective_mean=0.2500, "
        "positive_energy_mean=0.7500, negative_energy_mean=1.2500"
    )


def test_summary_defaults_optional_energy_metrics_to_zero():
    """Averaged energy diagnostics are optional in the objective outputs."""
    outputs = make_outputs()
    del outputs["positive_energy_mean"]
    del outputs["negative_energy_mean"]
    summary = summarize_objective(outputs)
    assert summary.endswith(
        "positive_energy_mean=0.0000, negative_energy_mean=0.0000"
    )
