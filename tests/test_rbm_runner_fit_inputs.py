"""Input validation for ``RBMRunner.fit`` in ``example/rbm_digits``.

``fit`` used to hand ``X`` straight to ``RestrictedBoltzmannMachine``/``SGD``.
Malformed data therefore failed deep inside sklearn (or, for NaN/Inf, not at
all: it reached the sampler and the gradient step).  These tests pin the new
up-front contract and the unchanged happy path.

``seaborn`` is only used for plotting (``plot_confusion_matrix``); the ``fit``
path never needs it, so the import helper below substitutes an empty module for
the duration of the import only and always restores ``sys.modules``.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "example" / "rbm_digits" / "rbm_digits.py"

SHAPE_MSG = "X must be a 2-D array with at least one sample and one feature"
FINITE_MSG = "X must contain only finite values"


@pytest.fixture(scope="module")
def rbm_digits():
    """Import the example without requiring the optional ``seaborn`` extra."""
    saved = sys.modules.get("seaborn")
    if saved is None:
        try:
            import seaborn  # noqa: F401  pylint: disable=unused-import
        except ModuleNotFoundError:
            # ``seaborn`` is declared in requirements/requirements_example.txt and is
            # only used by ``plot_confusion_matrix``; ``fit`` never touches it.
            sys.modules["seaborn"] = types.ModuleType("seaborn")
    try:
        spec = importlib.util.spec_from_file_location(
            "rbm_digits_under_test", MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if saved is None:
            sys.modules.pop("seaborn", None)
    return module


class _StubSampler:  # pylint: disable=too-few-public-methods
    """Deterministic sampler stand-in: no licence, no solver, no device."""

    def solve(self, ising_matrix):
        matrix = np.asarray(ising_matrix)
        return np.ones((1, matrix.shape[-1]), dtype=np.float32)


def _runner(module):
    runner = module.RBMRunner(
        n_components=4,
        learning_rate=0.1,
        batch_size=2,
        n_iter=1,
        verbose=False,
        plot_img=False,
        random_state=0,
    )
    runner.sampler = _StubSampler()
    return runner


def test_one_dimensional_input_is_rejected(rbm_digits):
    with pytest.raises(ValueError, match="2-D array"):
        _runner(rbm_digits).fit(np.zeros(8, dtype=np.float32))


def test_list_input_is_rejected_like_any_other_shape(rbm_digits):
    with pytest.raises(ValueError, match="2-D array"):
        _runner(rbm_digits).fit([0.0, 1.0, 2.0, 3.0])


def test_empty_sample_axis_is_rejected(rbm_digits):
    with pytest.raises(ValueError, match="at least one sample"):
        _runner(rbm_digits).fit(np.zeros((0, 8), dtype=np.float32))


def test_empty_feature_axis_is_rejected(rbm_digits):
    with pytest.raises(ValueError, match="at least one sample"):
        _runner(rbm_digits).fit(np.zeros((8, 0), dtype=np.float32))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_values_are_rejected(rbm_digits, bad):
    x = np.zeros((4, 4), dtype=np.float32)
    x[1, 2] = bad
    with pytest.raises(ValueError, match="finite values"):
        _runner(rbm_digits).fit(x)


def test_validation_happens_before_the_model_is_built(rbm_digits, monkeypatch):
    """A rejected matrix must never reach RBM/SGD construction."""
    calls = []

    def _explode(*args, **kwargs):  # pylint: disable=unused-argument
        calls.append(args)
        raise AssertionError("RestrictedBoltzmannMachine built for rejected input")

    monkeypatch.setattr(rbm_digits, "RestrictedBoltzmannMachine", _explode)
    with pytest.raises(ValueError, match="finite values"):
        _runner(rbm_digits).fit(np.full((4, 4), np.nan, dtype=np.float32))
    assert not calls


def test_valid_matrix_still_trains(rbm_digits):
    """The happy path is unchanged: one iteration on a small finite matrix."""
    x = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    runner = _runner(rbm_digits)

    result = runner.fit(x)

    assert result is runner
    assert runner.rbm is not None
    assert runner.rbm.num_visible == 4
    assert np.isfinite(runner.rbm.quadratic_coef.detach().numpy()).all()
