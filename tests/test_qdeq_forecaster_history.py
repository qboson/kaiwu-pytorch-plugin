"""DEQForecaster must reject histories that cannot be normalised per window."""

import sys
import types
from pathlib import Path

import numpy as np
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


def _model(module):
    return module.DEQForecaster(
        seq_len=96,
        pred_len=96,
        sampler=None,
        config=module.build_config(),
    ).eval()


def test_a_single_step_history_is_rejected(etth1):
    model = _model(etth1)
    history = torch.randn(2, 1, 7)

    with pytest.raises(ValueError, match="at least two time steps"):
        model(history)


def test_a_two_dimensional_history_is_rejected(etth1):
    model = _model(etth1)

    with pytest.raises(ValueError, match="batch, seq_len, variables"):
        model(torch.randn(2, 96))


def test_a_batched_history_still_forecasts(etth1):
    model = _model(etth1)
    history = torch.randn(2, 96, 7)

    with torch.no_grad():
        forecast = model(history)

    assert forecast.shape == (2, 96, 7)
    assert np.isfinite(forecast.numpy()).all()
