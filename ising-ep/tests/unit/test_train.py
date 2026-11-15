"""Tests for the two-phase EP training loop.

A fake solver stands in for Kaiwu so the pairing rules and the autograd
gradient can be checked exactly, without annealing randomness.
"""

# Test functions document intent in their summary; pytest fixtures and
# parametrised values do not benefit from Args sections.
# pylint: disable=missing-param-doc,missing-return-doc

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from examples.mnist_kaiwu.train import solve_phases, train_epoch
from ising_ep.model import IsingEPModel

from tests.conftest import FakeSolver, to_numpy

BETA = 0.5


class CountingSGD(torch.optim.SGD):
    """SGD that counts how many times ``step`` was called.

    Attributes:
        step_count: Number of completed optimizer steps.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Forward all arguments to ``SGD`` and start the counter at zero.

        Args:
            *args: Positional arguments for ``torch.optim.SGD``.
            **kwargs: Keyword arguments for ``torch.optim.SGD``.
        """
        super().__init__(*args, **kwargs)
        self.step_count = 0

    def step(self, closure: Any = None) -> Any:
        """Count the call and delegate to ``SGD.step``."""
        self.step_count += 1
        return super().step(closure)


def _model() -> IsingEPModel:
    """Return a 3-input, 2-hidden, 4-output model."""
    return IsingEPModel(n_in=3, hidden_sizes=2, n_classes=2, spins_per_class=2, seed=5)


FREE = np.array([1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
NUDGED = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0])


def test_solve_phases_shares_the_pair_seed_between_both_phases() -> None:
    """Free and nudged solves of one sample use the same seed."""
    solver = FakeSolver(FREE, NUDGED)

    solve_phases(_model(), solver, np.ones(3), _model().target_spins(0), BETA, 4321)

    assert [call["seed"] for call in solver.calls] == [4321, 4321]


def test_nudged_phase_warm_starts_from_the_free_solver_row() -> None:
    """The nudged solve receives the complete free solver row, aux included."""
    solver = FakeSolver(FREE, NUDGED)

    solve_phases(_model(), solver, np.ones(3), _model().target_spins(0), BETA, 7)

    assert solver.calls[0]["init_solution"] is None
    np.testing.assert_array_equal(
        solver.calls[1]["init_solution"], np.concatenate([FREE, [1.0]])
    )


def test_solve_phases_splits_states_on_the_hidden_boundary() -> None:
    """Decoded states are split into hidden and output blocks."""
    model = _model()
    solver = FakeSolver(FREE, NUDGED)

    pair = solve_phases(model, solver, np.ones(3), model.target_spins(0), BETA, 1)

    np.testing.assert_array_equal(to_numpy(pair.free_states[0]), FREE[:2])
    np.testing.assert_array_equal(to_numpy(pair.free_states[1]), FREE[2:])
    np.testing.assert_array_equal(to_numpy(pair.nudged_states[0]), NUDGED[:2])
    np.testing.assert_array_equal(to_numpy(pair.nudged_states[1]), NUDGED[2:])


def test_only_the_output_field_differs_between_the_paired_problems() -> None:
    """The nudge reaches the solver through the output field alone."""
    model = _model()
    solver = FakeSolver(FREE, NUDGED)

    solve_phases(model, solver, np.ones(3), model.target_spins(0), BETA, 1)

    free_q = to_numpy(solver.calls[0]["Q"])
    nudged_q = to_numpy(solver.calls[1]["Q"])
    # Hidden diagonal and the coupling block are untouched by beta.
    np.testing.assert_allclose(free_q[:2, :], nudged_q[:2, :])
    assert not np.allclose(np.diag(free_q)[2:], np.diag(nudged_q)[2:])


@pytest.mark.parametrize("beta", [0.0, -1.0])
def test_solve_phases_requires_a_strictly_positive_beta(beta: float) -> None:
    """A zero or negative nudge makes the EP gradient undefined."""
    model = _model()

    with pytest.raises(ValueError, match="strictly positive beta"):
        solve_phases(model, FakeSolver(FREE), np.ones(3), model.target_spins(0), beta, 1)


def test_train_epoch_steps_once_per_batch() -> None:
    """Every sample in a batch is solved before the single optimizer step."""
    model = _model()
    optimizer = CountingSGD(model.parameters(), lr=0.01)
    solver = FakeSolver(FREE, NUDGED)

    stats = train_epoch(
        model,
        optimizer,
        solver,
        np.ones((4, 3)),
        np.zeros(4, dtype=np.int64),
        order=[0, 1, 2, 3],
        beta=BETA,
        batch_size=2,
        pair_seed_fn=lambda index: 100 + index,
    )

    assert optimizer.step_count == 2
    assert stats["batch_updates"] == 2.0
    assert stats["n_samples"] == 4.0
    # Two solver calls per sample: one free, one nudged.
    assert len(solver.calls) == 8


def test_train_epoch_gradient_equals_the_two_phase_correlation_difference() -> None:
    """Autograd on ``(E_nudged - E_free) / beta`` reproduces the EP estimator."""
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    solver = FakeSolver(FREE, NUDGED)
    x = np.ones(3)

    train_epoch(
        model,
        optimizer,
        solver,
        x.reshape(1, 3),
        np.zeros(1, dtype=np.int64),
        order=[0],
        beta=BETA,
        batch_size=1,
        pair_seed_fn=lambda _index: 1,
    )

    h_free, o_free = FREE[:2], FREE[2:]
    h_nudged, o_nudged = NUDGED[:2], NUDGED[2:]
    np.testing.assert_allclose(
        to_numpy(model.weights[0].grad), (np.outer(x, h_free) - np.outer(x, h_nudged)) / BETA
    )
    np.testing.assert_allclose(
        to_numpy(model.weights[1].grad),
        (np.outer(h_free, o_free) - np.outer(h_nudged, o_nudged)) / BETA,
    )
    np.testing.assert_allclose(to_numpy(model.biases[0].grad), (h_free - h_nudged) / BETA)
    np.testing.assert_allclose(to_numpy(model.biases[1].grad), (o_free - o_nudged) / BETA)


def test_gradient_vanishes_when_both_phases_agree() -> None:
    """Identical free and nudged states carry no learning signal."""
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    solver = FakeSolver(FREE)
    before = model.parameter_snapshot()

    train_epoch(
        model,
        optimizer,
        solver,
        np.ones((1, 3)),
        np.zeros(1, dtype=np.int64),
        order=[0],
        beta=BETA,
        batch_size=1,
        pair_seed_fn=lambda _index: 1,
    )

    np.testing.assert_allclose(to_numpy(model.weights[0].grad), np.zeros((3, 2)))
    np.testing.assert_allclose(
        to_numpy(model.weights[0]), to_numpy(before["weights.0"])
    )


def test_train_epoch_reports_flip_statistics() -> None:
    """Flip rates summarise how far the nudge moved the solver state."""
    model = _model()
    solver = FakeSolver(FREE, NUDGED)

    stats = train_epoch(
        model,
        torch.optim.SGD(model.parameters(), lr=0.0),
        solver,
        np.ones((1, 3)),
        np.zeros(1, dtype=np.int64),
        order=[0],
        beta=BETA,
        batch_size=1,
        pair_seed_fn=lambda _index: 1,
    )

    assert stats["hidden_flip"] == pytest.approx(0.5)
    assert stats["output_flip"] == pytest.approx(0.5)
    assert stats["zero_update"] == pytest.approx(0.0)
    assert stats["hidden_zero_update"] == pytest.approx(0.0)
    assert "skip_nudge" not in stats


def test_stacked_layers_each_receive_their_correlation_difference() -> None:
    """Every weight in a two-hidden-layer net gets the two-phase estimator."""
    model = IsingEPModel(n_in=2, hidden_sizes=(2, 2), n_classes=2, spins_per_class=1, seed=2)
    free = np.array([1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
    nudged = np.array([-1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
    x = np.array([0.5, -0.25])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    train_epoch(
        model,
        optimizer,
        FakeSolver(free, nudged),
        x.reshape(1, 2),
        np.zeros(1, dtype=np.int64),
        order=[0],
        beta=BETA,
        batch_size=1,
        pair_seed_fn=lambda _index: 1,
    )

    free_layers = [free[:2], free[2:4], free[4:]]
    nudged_layers = [nudged[:2], nudged[2:4], nudged[4:]]
    previous_free = x
    previous_nudged = x
    for index, (free_layer, nudged_layer) in enumerate(zip(free_layers, nudged_layers)):
        expected = (
            np.outer(previous_free, free_layer) - np.outer(previous_nudged, nudged_layer)
        ) / BETA
        np.testing.assert_allclose(to_numpy(model.weights[index].grad), expected)
        previous_free = free_layer
        previous_nudged = nudged_layer
