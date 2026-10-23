"""Warmup-cosine schedule contract for the BM generation example."""

import math
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "bm_generation"))

from trainer import CosineScheduleWithWarmup  # noqa: E402


def build_schedule():
    """Builds the scheduler around a one-parameter optimizer with lr=1."""
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    return CosineScheduleWithWarmup(
        optimizer, num_warmup_steps=4, num_training_steps=12
    )


def test_warmup_is_linear_and_reaches_one():
    """Warmup ramps from 0 to 1 across the warmup steps."""
    schedule = build_schedule()
    assert schedule.lr_lambda(0) == pytest.approx(0.0)
    assert schedule.lr_lambda(2) == pytest.approx(0.5)
    assert schedule.lr_lambda(3) == pytest.approx(0.75)


def test_descent_follows_the_half_cosine():
    """After warmup the factor follows ``0.5 * (1 + cos(2π * cycles * progress))``."""
    schedule = build_schedule()
    assert schedule.lr_lambda(4) == pytest.approx(1.0)
    assert schedule.lr_lambda(8) == pytest.approx(0.5)
    assert schedule.lr_lambda(12) == pytest.approx(0.0)


def test_scheduler_steps_scale_the_optimizer_learning_rate():
    """Stepping the scheduler applies the factor to the optimizer's lr."""
    schedule = build_schedule()
    optimizer = schedule.optimizer
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)

    def advance():
        optimizer.step()
        schedule.step()

    advance()
    assert schedule.last_epoch == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.25)
    for _ in range(3):
        advance()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)
    for _ in range(4):
        advance()
    assert schedule.last_epoch == 8
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
