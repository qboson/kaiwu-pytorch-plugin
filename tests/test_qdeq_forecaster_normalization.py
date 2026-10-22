"""DEQForecaster must map its forecast back through the per-window statistics."""

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


class ZeroEquilibrium(torch.nn.Module):
    """Returns zero tokens for any input."""

    def forward(self, tokens, z0=None):
        """Returns zeros shaped like the input tokens."""
        del z0
        return torch.zeros_like(tokens)


def test_forecast_is_denormalized_with_the_window_statistics(etth1):
    model = etth1.DEQForecaster(96, 96, sampler=None, config=etth1.build_config()).eval()
    model.deq = ZeroEquilibrium()
    for parameter in model.head.parameters():
        parameter.data.zero_()
    history = torch.randn(2, 96, 3)

    with torch.no_grad():
        forecast = model(history)

    expected = history.mean(dim=1, keepdim=True).expand(-1, 96, -1)
    assert forecast.shape == (2, 96, 3)
    torch.testing.assert_close(forecast, expected)
