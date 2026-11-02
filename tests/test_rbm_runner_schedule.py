"""Contract test for RBMRunner training-schedule validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "example" / "rbm_digits" / "rbm_digits.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("rbm_runner_schedule_uut", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kwargs", [{"n_iter": 0}, {"n_iter": -1}, {"batch_size": 0}, {"batch_size": -2}])
def test_rejects_a_non_positive_training_schedule(kwargs):
    module = _load_runner()

    with pytest.raises(ValueError, match="positive"):
        module.RBMRunner(n_components=2, **kwargs)


@pytest.mark.parametrize("rate", [0.0, -0.1, float("nan"), float("inf")])
def test_rejects_a_non_positive_or_non_finite_learning_rate(rate):
    module = _load_runner()

    with pytest.raises(ValueError, match="learning_rate"):
        module.RBMRunner(n_components=2, learning_rate=rate)
