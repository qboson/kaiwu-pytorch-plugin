"""Tests for free-phase-only evaluation.

The central property is that no label reaches the solver, so a trained model is
scored on the inputs alone.
"""

# Test functions document intent in their summary; pytest fixtures and
# parametrised values do not benefit from Args sections.
# pylint: disable=missing-param-doc,missing-return-doc

from __future__ import annotations

import numpy as np
import pytest

from examples.mnist_kaiwu.evaluate import evaluate_free_only
from ising_ep.model import IsingEPModel

from tests.conftest import FakeSolver, to_numpy


def _model() -> IsingEPModel:
    """Return a 3-input, 2-hidden, 2-class model with 2 spins per class."""
    return IsingEPModel(n_in=3, hidden_sizes=2, n_classes=2, spins_per_class=2, seed=5)


# Output block sums are [2, -2], so class 0 wins.
PREDICTS_CLASS_0 = np.array([1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
# Output block sums are [-2, 2], so class 1 wins.
PREDICTS_CLASS_1 = np.array([1.0, -1.0, -1.0, -1.0, 1.0, 1.0])


def test_evaluation_uses_one_free_solve_per_image() -> None:
    """Exactly one solver call is made per test image, never a nudged pair."""
    solver = FakeSolver(PREDICTS_CLASS_0)

    evaluate_free_only(_model(), solver, np.ones((5, 3)), np.zeros(5, dtype=np.int64), eval_seed=0)

    assert len(solver.calls) == 5
    assert all(call["init_solution"] is None for call in solver.calls)


def test_evaluation_never_puts_the_label_into_the_solver_problem() -> None:
    """Two images with different labels but equal pixels give the same QUBO."""
    solver = FakeSolver(PREDICTS_CLASS_0)

    evaluate_free_only(
        _model(), solver, np.ones((2, 3)), np.array([0, 1], dtype=np.int64), eval_seed=0
    )

    np.testing.assert_array_equal(
        to_numpy(solver.calls[0]["Q"]), to_numpy(solver.calls[1]["Q"])
    )


def test_each_image_gets_its_own_reproducible_seed() -> None:
    """Sample ``i`` anneals with ``eval_seed + i``."""
    solver = FakeSolver(PREDICTS_CLASS_0)

    evaluate_free_only(
        _model(), solver, np.ones((3, 3)), np.zeros(3, dtype=np.int64), eval_seed=900
    )

    assert [call["seed"] for call in solver.calls] == [900, 901, 902]


def test_metrics_and_records_describe_every_prediction() -> None:
    """Accuracy and the per-sample records agree with the scripted states."""
    labels = np.array([0, 1], dtype=np.int64)
    solver = FakeSolver(PREDICTS_CLASS_0, PREDICTS_CLASS_1)

    metrics, records = evaluate_free_only(
        _model(), solver, np.ones((2, 3)), labels, eval_seed=0
    )

    assert metrics["n"] == 2
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert [record["prediction"] for record in records] == [0, 1]
    assert [record["label"] for record in records] == [0, 1]
    assert [record["index"] for record in records] == [0, 1]


def test_wrong_predictions_lower_the_accuracy() -> None:
    """A model that always answers class 0 scores half on a balanced pair."""
    labels = np.array([0, 1], dtype=np.int64)
    solver = FakeSolver(PREDICTS_CLASS_0)

    metrics, _records = evaluate_free_only(
        _model(), solver, np.ones((2, 3)), labels, eval_seed=0
    )

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["per_class"][0]["accuracy"] == pytest.approx(1.0)
    assert metrics["per_class"][1]["accuracy"] == pytest.approx(0.0)
