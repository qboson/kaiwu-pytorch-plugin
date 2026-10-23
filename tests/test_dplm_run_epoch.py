"""Aggregation contract for one DPLM workflow epoch."""

import os
import sys

import pytest
import torch

pytest.importorskip("tqdm")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from workflows.workflow_helpers import run_epoch  # noqa: E402


BATCHES = [
    {"targets": torch.tensor([1.0])},
    {"targets": torch.tensor([3.0, 3.0, 3.0])},
]


class StubGenerator(torch.nn.Module):
    """Objective stub whose loss is the batch mean scaled by one parameter."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.grad_enabled = []

    def objective(self, payload):
        self.grad_enabled.append(torch.is_grad_enabled())
        base = payload["targets"].mean()
        return {
            "energy_objective": self.weight * base,
            "positive_energy_mean": base + 1.0,
        }


class StubOptimizer:
    """Counts the optimizer calls made by the training branch."""

    def __init__(self):
        self.steps = 0
        self.zeroed = 0

    def zero_grad(self, set_to_none=True):
        self.zeroed += 1

    def step(self):
        self.steps += 1


def test_metrics_are_weighted_by_batch_size():
    """Each batch contributes in proportion to its number of examples."""
    generator = StubGenerator()
    metrics = run_epoch(
        generator, BATCHES, optimizer=None, grad_clip_norm=0.0, description="eval"
    )
    # (1 * 2 + 3 * 6) / 4 = 5 and (1 * 2 + 3 * 4) / 4 = 3.5
    assert metrics["energy_objective_mean"] == pytest.approx(5.0)
    assert metrics["positive_energy_mean"] == pytest.approx(3.5)


def test_evaluation_runs_without_grad_and_never_steps():
    """Without an optimizer the loader is read under ``torch.no_grad``."""
    generator = StubGenerator()
    run_epoch(generator, BATCHES, optimizer=None, grad_clip_norm=0.0, description="eval")
    assert generator.grad_enabled == [False, False]
    assert generator.training is False


def test_training_steps_once_per_batch_and_backpropagates():
    """The training branch zeroes, clips and steps on every batch."""
    generator = StubGenerator()
    optimizer = StubOptimizer()
    run_epoch(generator, BATCHES, optimizer=optimizer, grad_clip_norm=0.5, description="train")
    assert generator.grad_enabled == [True, True]
    assert optimizer.steps == 2
    assert optimizer.zeroed == 2
    assert generator.weight.grad is not None
    assert torch.isfinite(generator.weight.grad).all()


def test_empty_loader_reports_zero_instead_of_dividing_by_zero():
    """An empty epoch returns a zero mean rather than raising."""
    generator = StubGenerator()
    metrics = run_epoch(
        generator, [], optimizer=None, grad_clip_norm=0.0, description="eval"
    )
    assert metrics == {"energy_objective_mean": 0.0}
