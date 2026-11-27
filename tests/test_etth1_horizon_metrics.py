"""Forecast lead metrics use every target, including a short final batch."""

import csv
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def modules(monkeypatch):
    pytest.importorskip("pandas", exc_type=ImportError)
    folder = Path(__file__).resolve().parents[1] / "example" / "qdeq" / "simple"
    monkeypatch.syspath_prepend(str(folder))
    names = ("etth1_common", "simple_forecast_example")
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield tuple(importlib.import_module(name) for name in names)
    for name in names:
        sys.modules.pop(name, None)


class Forecast(torch.nn.Module):
    def __init__(self, *_args):
        super().__init__()
        self.deq = SimpleNamespace(get_last_stats=lambda: {"forward_nstep": 2})

    def forward(self, history):
        return history


def make_loader():
    errors = torch.tensor([[[1., 1.], [0., 0.]], [[3., -1.], [2., 0.]], [[-9., 5.], [6., 4.]]])
    return DataLoader(TensorDataset(errors, torch.zeros_like(errors)), batch_size=2)


def test_metrics_are_per_lead_and_weight_the_short_batch(modules):
    common, _ = modules
    model = Forecast()
    aggregate = common.evaluate(model, make_loader())
    assert set(aggregate) == {"mse", "mae", "steps"}
    detail = common.evaluate(model, make_loader(), by_horizon=True)
    assert detail["mse_by_horizon"] == pytest.approx([118 / 6, 56 / 6])
    assert detail["mae_by_horizon"] == pytest.approx([20 / 6, 12 / 6])
    assert detail["mse"] == pytest.approx(aggregate["mse"])


def test_forecast_cli_writes_the_ordered_horizon_csv(modules, monkeypatch, tmp_path):
    _, cli = modules
    checkpoint = tmp_path / "weights.pt"
    output = tmp_path / "reports" / "horizons.csv"
    torch.save({"state_dict": Forecast().state_dict()}, checkpoint)
    monkeypatch.setattr(cli, "build_loaders", lambda *_args, **_options: {"test": make_loader()})
    monkeypatch.setattr(cli, "DEQForecaster", Forecast)
    monkeypatch.setattr(sys, "argv", [
        "simple_forecast_example.py", "--sampler", "anderson", "--checkpoint", str(checkpoint),
        "--horizon-metrics", str(output),
    ])

    cli.main()

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["lead_hours"]) for row in rows] == [1, 2]
    assert [float(row["mse"]) for row in rows] == pytest.approx([118 / 6, 56 / 6])
    assert [float(row["mae"]) for row in rows] == pytest.approx([20 / 6, 12 / 6])
