"""Contract test for RBMRunner.fit finite-input validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "example" / "rbm_digits" / "rbm_digits.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("rbm_runner_finite_uut", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "data",
    [
        np.array([[0.0, float("nan")], [1.0, 2.0]]),
        np.array([[0.0, float("inf")], [1.0, 2.0]]),
    ],
)
def test_rejects_non_finite_training_data(data):
    module = _load_runner()
    runner = module.RBMRunner(n_components=2, batch_size=1, n_iter=1)

    with pytest.raises(ValueError, match="finite"):
        runner.fit(data)
