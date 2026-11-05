# -*- coding: utf-8 -*-
"""BM generation cost weights must be finite and non-negative."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest

TRAINER = Path(__file__).resolve().parents[1] / "example/bm_generation/trainer.py"
SPEC = importlib.util.spec_from_file_location("bm_trainer_cost", TRAINER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _trainer():
    return MODULE.Trainer([], Mock(), Mock(), num_visible=2, num_hidden=1)


@pytest.mark.parametrize(
    "alpha,beta",
    [(-0.1, 1.0), (0.5, -1.0), (float("nan"), 1.0), (0.5, float("inf"))],
)
def test_invalid_cost_weights_are_rejected(alpha: float, beta: float) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        _trainer().set_cost_parameter(alpha, beta)


def test_valid_cost_weights_are_stored_as_floats() -> None:
    trainer = _trainer()

    trainer.set_cost_parameter(0.5, 10)

    assert trainer.cost_param == {"alpha": 0.5, "beta": 10.0}
