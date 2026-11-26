"""Check that the QDEQ forecast preview comes from the scored test batch."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class BatchDependentForecaster(nn.Module):
    """Return the batch size as every forecast value."""

    def __init__(self):
        super().__init__()
        self.batch_sizes = []
        self.deq = SimpleNamespace(get_last_stats=lambda: {"forward_nstep": 1})

    def forward(self, history):
        self.batch_sizes.append(len(history))
        return torch.full_like(history, len(history))


def load_module(path, name):
    """Load an example module without changing the import search path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_forecast_preview_reuses_the_scored_prediction(monkeypatch, tmp_path, capsys):
    """The first prediction depends on whether its batch has one or two rows."""
    pytest.importorskip("pandas")
    examples = Path(__file__).resolve().parents[1] / "example" / "qdeq" / "simple"
    common = load_module(examples / "etth1_common.py", "etth1_common")
    monkeypatch.setitem(sys.modules, "etth1_common", common)
    forecast = load_module(examples / "simple_forecast_example.py", "qdeq_forecast_example")

    model = BatchDependentForecaster()
    data = torch.zeros(3, 8, 1)
    test_loader = DataLoader(TensorDataset(data, data), batch_size=2)
    checkpoint = tmp_path / "forecast.pt"
    torch.save({"state_dict": model.state_dict()}, checkpoint)

    monkeypatch.setattr(forecast, "build_loaders", lambda _path: {"test": test_loader})
    monkeypatch.setattr(forecast, "DEQForecaster", lambda *_args: model)
    monkeypatch.setattr(
        sys, "argv", ["simple_forecast_example.py", "--sampler", "anderson", "--checkpoint", str(checkpoint)]
    )
    forecast.main()

    output = capsys.readouterr().out
    assert "test mse 3.0000 mae 1.6667" in output
    assert f"  forecast {[2.0] * 8}" in output
    assert f"  actual   {[0.0] * 8}" in output
    assert model.batch_sizes == [2, 1]
