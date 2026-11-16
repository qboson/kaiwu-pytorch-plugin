"""QDEQ's forecaster should handle one-hour histories and ordinary windows."""

import pytest
import torch

pytest.importorskip("pandas")

from example.qdeq.simple.etth1_common import DEQForecaster, QDEQConfig


@pytest.mark.parametrize("lookback", [1, 4])
def test_forecaster_uses_finite_window_scale(lookback):
    model = DEQForecaster(lookback, 2, None, QDEQConfig(f_thres=3), d_model=4, n_heads=1)
    model.eval()
    history = torch.arange(1., 1. + 2 * lookback).reshape(1, lookback, 2)

    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(2.)
        forecast = model(history)

    scale = torch.zeros_like(history[:, :1]) if lookback == 1 else history.std(dim=1, keepdim=True)
    expected = history.mean(dim=1, keepdim=True) + 2 * (scale + 1e-5)
    torch.testing.assert_close(forecast, expected.expand(1, 2, 2))
