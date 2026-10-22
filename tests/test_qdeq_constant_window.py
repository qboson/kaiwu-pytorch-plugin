"""A constant history window must not produce NaN forecasts."""

import sys
import types
from pathlib import Path

import pytest
import torch

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


def test_a_constant_window_stays_finite(etth1):
    model = etth1.DEQForecaster(96, 96, sampler=None, config=etth1.build_config()).eval()
    history = torch.full((2, 96, 3), 5.0)

    with torch.no_grad():
        forecast = model(history)

    assert forecast.shape == (2, 96, 3)
    assert torch.isfinite(forecast).all()

