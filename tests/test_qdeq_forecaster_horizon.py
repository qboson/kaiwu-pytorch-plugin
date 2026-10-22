"""DEQForecaster must use its constructor look-back and horizon."""

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


def test_a_non_standard_look_back_and_horizon_keep_their_shape(etth1):
    model = etth1.DEQForecaster(
        seq_len=48,
        pred_len=24,
        sampler=None,
        config=etth1.build_config(),
    ).eval()
    history = torch.randn(2, 48, 3)

    with torch.no_grad():
        forecast = model(history)

    assert forecast.shape == (2, 24, 3)
    assert np.isfinite(forecast.numpy()).all()
