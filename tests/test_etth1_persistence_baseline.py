"""ETTh1 persistence is evaluated on the same windows as learned forecasters."""

import importlib
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

pytest.importorskip("pandas")


@pytest.fixture
def forecast_cli(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "example" / "qdeq" / "simple"
    monkeypatch.syspath_prepend(str(root))
    for name in ("simple_forecast_example", "etth1_common"):
        monkeypatch.setitem(sys.modules, name, None)
        sys.modules.pop(name)
    return importlib.import_module("simple_forecast_example")


def test_persistence_cli_reads_no_checkpoint_and_uses_the_last_observation(
    forecast_cli, monkeypatch, tmp_path, capsys
):
    history = torch.tensor([[[9.0, 12.0], [2.0, -3.0]], [[7.0, 7.0], [-1.0, 5.0]]])
    target = (history[:, -1:, :] + torch.tensor([1.0, -3.0])).expand(-1, 96, -1)
    loader = DataLoader(TensorDataset(history, target), batch_size=1)
    monkeypatch.setattr(forecast_cli, "build_loaders", lambda *args, **kwargs: {"test": loader})

    def unexpected(*args, **kwargs):
        pytest.fail("Persistence must not load weights or construct a sampler")

    monkeypatch.setattr(forecast_cli, "build_sampler", unexpected)
    monkeypatch.setattr(torch, "load", unexpected)
    monkeypatch.setattr(sys, "argv", [
        "simple_forecast_example.py", "--persistence", "--checkpoint", str(tmp_path / "unused.pt")
    ])
    forecast_cli.main()

    output = capsys.readouterr().out
    assert "persistence: test mse 5.0000 mae 2.0000 (mean forward steps 0.0)" in output
    assert "forecast [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0]" in output
