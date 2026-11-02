"""Contract test for DBNTrainer training-data validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "example" / "dbn_digits" / "dbn_trainer.py"


def _load_trainer():
    spec = importlib.util.spec_from_file_location("dbn_trainer_data_uut", TRAINER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeDBN:
    pass


@pytest.mark.parametrize("data", [np.zeros(4), np.zeros((0, 4)), np.zeros((4, 0))])
def test_rejects_non_training_matrices(data):
    module = _load_trainer()
    monkeypatch_target = (module.UnsupervisedDBN,)
    dbn = type("_DBN", monkeypatch_target, {})()
    trainer = module.DBNTrainer(verbose=False)

    with pytest.raises(ValueError, match="2-D"):
        trainer.train(dbn, data)
