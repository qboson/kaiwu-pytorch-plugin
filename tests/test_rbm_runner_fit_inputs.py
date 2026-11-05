"""Contract test for RBMRunner.fit input validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "example" / "rbm_digits" / "rbm_digits.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("rbm_runner_fit_inputs_uut", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("data", [np.zeros(4), np.zeros((0, 4)), np.zeros((4, 0))])
def test_fit_rejects_non_training_matrices(data):
    module = _load_runner()
    runner = module.RBMRunner(n_components=2, batch_size=2, n_iter=1)

    with pytest.raises(ValueError, match="2-D"):
        runner.fit(data)
