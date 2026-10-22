"""DEQForecaster must reject histories whose length differs from the built model."""

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


def test_a_history_of_another_length_is_rejected(etth1):
    model = _model(etth1)

    with pytest.raises(ValueError, match="time steps"):
        model(torch.randn(2, 48, 7))


def test_the_built_look_back_still_forecasts(etth1):
    model = _model(etth1)

    with torch.no_grad():
        forecast = model(torch.randn(2, 96, 7))

    assert forecast.shape == (2, 96, 7)
    assert np.isfinite(forecast.numpy()).all()
