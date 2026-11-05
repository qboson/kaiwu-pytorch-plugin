"""Contract test for DBNTrainer RBM learning-rate validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "example" / "dbn_digits" / "dbn_trainer.py"


def _load_trainer():
    spec = importlib.util.spec_from_file_location("dbn_trainer_lr_uut", TRAINER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("rate", [0.0, -0.1, float("nan"), float("inf")])
def test_rejects_a_non_positive_or_non_finite_learning_rate(rate):
    module = _load_trainer()

    with pytest.raises(ValueError, match="learning_rate_rbm"):
        module.DBNTrainer(learning_rate_rbm=rate)
