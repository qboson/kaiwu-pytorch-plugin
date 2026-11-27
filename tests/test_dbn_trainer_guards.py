"""Entry-point guards for ``DBNTrainer`` in ``example/dbn_digits``.

``_train_rbm_layer`` assumed the DataLoader always yields at least one batch.
With ``drop_last=True`` and fewer samples than ``batch_size`` the loader is
empty, so ``avg_loss`` stayed unassigned and the verbose epoch statistics
raised ``UnboundLocalError`` (with ``verbose=False`` the epoch silently did
nothing).  These tests pin the diagnosable error and the untouched happy path.

``n_epochs_rbm=0`` keeps its initialization-only meaning: it must stay legal
even when a ``drop_last`` loader would be empty, because no training happens.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "example" / "dbn_digits"))

from dbn_trainer import DBNTrainer  # noqa: E402  pylint: disable=wrong-import-position
from kaiwu.torch_plugin import (  # noqa: E402  pylint: disable=wrong-import-position
    RestrictedBoltzmannMachine,
)


class _StubSampler:  # pylint: disable=too-few-public-methods
    """Deterministic sampler stand-in: no licence, no solver, no device."""

    def solve(self, ising_matrix):
        matrix = np.asarray(ising_matrix)
        return np.ones((1, matrix.shape[-1]), dtype=np.float32)


def _rbm():
    return RestrictedBoltzmannMachine(num_visible=4, num_hidden=3)


def _data(n_samples=4):
    return np.linspace(0.0, 1.0, n_samples * 4, dtype=np.float32).reshape(n_samples, 4)


def test_drop_last_empty_loader_is_rejected_with_a_diagnosable_error():
    trainer = DBNTrainer(
        n_epochs_rbm=1,
        batch_size=100,
        drop_last=True,
        shuffle=False,
        verbose=True,
        plot_img=False,
    )
    with pytest.raises(ValueError) as excinfo:
        trainer._train_rbm_layer(  # pylint: disable=protected-access
            _rbm(), _data(4), 0
        )

    message = str(excinfo.value)
    assert "produced no batches" in message
    assert "n_samples=4" in message
    assert "batch_size=100" in message
    assert "drop_last=True" in message
    assert "drop_last=False" in message  # the remedy it suggests


def test_silent_no_op_is_rejected_even_without_verbose():
    """``verbose=False`` used to swallow the empty loader entirely."""
    trainer = DBNTrainer(
        n_epochs_rbm=1,
        batch_size=100,
        drop_last=True,
        shuffle=False,
        verbose=False,
        plot_img=False,
    )
    with pytest.raises(ValueError, match="produced no batches"):
        trainer._train_rbm_layer(  # pylint: disable=protected-access
            _rbm(), _data(4), 0
        )


def test_zero_epochs_stays_initialization_only():
    """PR #1863 relies on ``n_epochs_rbm=0`` working with an empty loader."""
    trainer = DBNTrainer(
        n_epochs_rbm=0,
        batch_size=100,
        drop_last=True,
        shuffle=False,
        verbose=True,
        plot_img=False,
    )
    hidden = trainer._train_rbm_layer(  # pylint: disable=protected-access
        _rbm(), _data(4), 0
    )

    assert hidden.shape == (4, 3)
    assert np.isfinite(hidden).all()


def test_non_empty_loader_still_trains():
    trainer = DBNTrainer(
        n_epochs_rbm=1,
        batch_size=2,
        drop_last=False,
        shuffle=False,
        verbose=True,
        plot_img=False,
    )
    trainer.sampler = _StubSampler()

    hidden = trainer._train_rbm_layer(  # pylint: disable=protected-access
        _rbm(), _data(4), 0
    )

    assert hidden.shape == (4, 3)
    assert np.isfinite(hidden).all()


@pytest.mark.parametrize("n_epochs", [-1, -10])
def test_negative_epochs_are_rejected(n_epochs):
    with pytest.raises(ValueError) as excinfo:
        DBNTrainer(n_epochs_rbm=n_epochs)

    message = str(excinfo.value)
    assert "n_epochs_rbm" in message
    assert ">= 0" in message
    assert repr(n_epochs) in message


@pytest.mark.parametrize(
    "learning_rate", [0, 0.0, -0.1, float("nan"), float("inf"), float("-inf")]
)
def test_non_positive_or_non_finite_learning_rate_is_rejected(learning_rate):
    with pytest.raises(ValueError) as excinfo:
        DBNTrainer(learning_rate_rbm=learning_rate)

    message = str(excinfo.value)
    assert "learning_rate_rbm" in message
    assert repr(learning_rate) in message


def test_default_configuration_is_still_accepted():
    trainer = DBNTrainer()

    assert trainer.n_epochs_rbm == 10
    assert trainer.learning_rate_rbm == 0.1
    assert trainer.drop_last is False
