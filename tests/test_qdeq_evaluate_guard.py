"""evaluate must reject a loader that yields no samples."""

import sys
import types
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "qdeq" / "simple"


@pytest.fixture
def etth1(monkeypatch):
    """Imports etth1_common with a stubbed pandas module."""
    pandas = types.ModuleType("pandas")
    pandas.read_csv = lambda path: None
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    sys.modules.pop("etth1_common", None)
    import etth1_common

    yield etth1_common
    sys.modules.pop("etth1_common", None)


class FakeDeq:
    """Reports one solver step for every forward."""

    def get_last_stats(self):
        """Returns the minimal stats evaluate reads."""
        return {"forward_nstep": 1}


class FakeModel:
    """Identity forecaster that records evaluate calls."""

    def __init__(self):
        self.deq = FakeDeq()

    def eval(self):
        """Marks evaluation mode."""
        return self

    def __call__(self, history):
        """Returns the history unchanged."""
        return history


def test_an_empty_loader_is_rejected(etth1):
    loader = DataLoader(
        TensorDataset(torch.zeros(0, 96, 7), torch.zeros(0, 96, 7)), batch_size=4
    )

    with pytest.raises(ValueError, match="at least one sample"):
        etth1.evaluate(FakeModel(), loader)


def test_a_non_empty_loader_reports_finite_metrics(etth1):
    targets = torch.zeros(6, 96, 7)
    loader = DataLoader(TensorDataset(targets, targets), batch_size=4)

    metrics = etth1.evaluate(FakeModel(), loader)

    assert metrics["mse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["steps"] == 1.0
