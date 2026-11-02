"""Contract test for DBNTrainer with an empty drop_last loader."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "example" / "dbn_digits" / "dbn_trainer.py"


def _load_trainer():
    spec = importlib.util.spec_from_file_location("dbn_trainer_empty_uut", TRAINER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeRBM:
    device = "cpu"

    def parameters(self):
        return iter(())


def test_empty_drop_last_loader_raises_a_clear_value_error(monkeypatch):
    module = _load_trainer()
    monkeypatch.setattr(module, "SGD", lambda parameters, lr: object())
    trainer = module.DBNTrainer(batch_size=10, drop_last=True, verbose=True)

    with pytest.raises(ValueError, match="batch"):
        trainer._train_rbm_layer(_FakeRBM(), np.zeros((4, 4), dtype=np.float32), 0)
