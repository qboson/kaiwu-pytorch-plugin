"""The qvae_cell trainer must reject budgets that cannot form loaders or epochs."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def trainer_module(monkeypatch):
    """Import the example trainer without leaking its local ``models`` module."""
    saved = {name: sys.modules.pop(name) for name in ("trainer", "models") if name in sys.modules}
    monkeypatch.syspath_prepend(str(ROOT / "example" / "qvae_cell"))
    try:
        import trainer  # pylint: disable=import-outside-toplevel

        yield trainer
    finally:
        sys.modules.pop("trainer", None)
        sys.modules.pop("models", None)
        sys.modules.update(saved)


def make_args(**overrides):
    values = {"val_percentage": 0.1, "batch_size": 8, "epochs": 2}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_valid_budgets_are_stored(trainer_module):
    args = make_args()

    trainer = trainer_module.Trainer(args, "cpu")

    assert trainer.args is args
    assert trainer.device == "cpu"


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5])
def test_val_percentage_must_leave_both_splits_nonempty(trainer_module, value):
    with pytest.raises(ValueError, match="val_percentage"):
        trainer_module.Trainer(make_args(val_percentage=value), "cpu")


@pytest.mark.parametrize("value", [0, -4])
def test_batch_size_must_be_positive(trainer_module, value):
    with pytest.raises(ValueError, match="batch_size"):
        trainer_module.Trainer(make_args(batch_size=value), "cpu")


@pytest.mark.parametrize("value", [0, -2])
def test_epochs_must_be_positive(trainer_module, value):
    with pytest.raises(ValueError, match="epochs"):
        trainer_module.Trainer(make_args(epochs=value), "cpu")
