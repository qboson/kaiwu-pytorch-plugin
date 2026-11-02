"""Contract test for DBNTrainer training-epoch validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "example" / "dbn_digits" / "dbn_trainer.py"


def _load_trainer():
    spec = importlib.util.spec_from_file_location("dbn_trainer_epochs_uut", TRAINER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("epochs", [0, -1])
def test_rejects_a_non_positive_training_schedule(epochs):
    module = _load_trainer()

    with pytest.raises(ValueError, match="n_epochs_rbm"):
        module.DBNTrainer(n_epochs_rbm=epochs)
