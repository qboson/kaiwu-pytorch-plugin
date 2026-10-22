# -*- coding: utf-8 -*-
"""Tests for the learning-direction example."""

from __future__ import annotations

import itertools
import os
import subprocess
import sys

import pytest
import torch

EXAMPLE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../example/learning_direction")
)
if EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, EXAMPLE_ROOT)

from learning_direction import (  # noqa: E402  # pylint: disable=wrong-import-position
    evaluate_learning,
    exact_pattern_probability,
    train_steps,
    RestrictedBoltzmannMachine,
)


def test_exact_visible_probabilities_sum_to_one() -> None:
    """Enumerating all visible patterns recovers the normalised marginal."""
    torch.manual_seed(0)
    model = RestrictedBoltzmannMachine(num_visible=3, num_hidden=2)
    total = sum(
        exact_pattern_probability(model, torch.tensor(bits, dtype=torch.float32))
        for bits in itertools.product([0.0, 1.0], repeat=3)
    )
    assert abs(total - 1.0) < 1e-6


def test_documented_loop_increases_the_pattern_probability() -> None:
    """One run of the loop raises p(data) and pushes the patterns below average."""
    result = evaluate_learning(steps=200, seed=0)

    assert result["p_after"] > result["p_before"]
    assert result["gap_after"] > result["gap_before"]


def test_the_check_is_directional() -> None:
    """Negating the objective must lower p(data), proving the guard has teeth."""
    normal = evaluate_learning(steps=200, seed=0)
    flipped = evaluate_learning(steps=200, seed=0, flip_sign=True)

    assert normal["p_after"] > normal["p_before"]
    assert flipped["p_after"] < flipped["p_before"]


def test_train_steps_rejects_zero() -> None:
    """A non-positive step count fails early."""
    model = RestrictedBoltzmannMachine(num_visible=3, num_hidden=2)
    with pytest.raises(ValueError):
        train_steps(model, steps=0, learning_rate=0.4)


def test_cli_entry_point_passes() -> None:
    """The documented command reports the measured improvement and exits 0."""
    script = os.path.join(EXAMPLE_ROOT, "learning_direction.py")
    proc = subprocess.run(
        [sys.executable, script, "--steps", "200"],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=EXAMPLE_ROOT,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "PASS" in proc.stdout
    assert "p(pattern)" in proc.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
