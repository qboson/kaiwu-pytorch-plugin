"""ETTh1 forecast preprocessing follows the saved training statistics."""

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def scripts(monkeypatch):
    pytest.importorskip("pandas", exc_type=ImportError)
    directory = Path(__file__).resolve().parents[1] / "example" / "qdeq" / "simple"
    module_names = ("etth1_common", "simple_train_example", "simple_forecast_example")
    monkeypatch.syspath_prepend(str(directory))
    for name in module_names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    modules = tuple(importlib.import_module(name) for name in module_names)
    common = modules[0]
    monkeypatch.setattr(common, "TRAIN_ROWS", 6)
    monkeypatch.setattr(common, "VAL_ROWS", 4)
    monkeypatch.setattr(common, "TEST_ROWS", 4)
    yield modules
    for name in module_names:
        sys.modules.pop(name, None)


def write_csv(path, reordered=False, missing=False, shifted=False):
    import pandas as pd

    index = np.arange(14, dtype=np.float32)
    frame = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=14, freq="h"),
        "OT": index,
        "aux": 2 * index,
    })
    if shifted:
        frame.loc[:5, "OT"] += 100
        frame.loc[:5, "aux"] -= 500
    if reordered:
        frame = frame[["date", "aux", "OT"]]
    if missing:
        frame = frame.drop(columns="aux")
    frame.to_csv(path, index=False)


class TinyForecaster(torch.nn.Module):
    def __init__(self, *_args):
        super().__init__()
        self.factor = torch.nn.Parameter(torch.tensor(1.0))
        self.deq = SimpleNamespace(
            get_last_stats=lambda: {"forward_nstep": 1, "backward_nstep": 1}
        )

    def forward(self, history):
        return history * self.factor


def test_training_and_forecasting_share_saved_scaling(scripts, monkeypatch, tmp_path, capsys):
    common, train, forecast = scripts
    train_path = tmp_path / "original.csv"
    forecast_path = tmp_path / "reordered.csv"
    checkpoint = tmp_path / "model.pt"
    write_csv(train_path)
    write_csv(forecast_path, reordered=True, shifted=True)
    args = SimpleNamespace(
        data_path=train_path, checkpoint=checkpoint, sampler="anderson", seed=0,
        lr=0.0, epochs=1, max_train_batches=1,
    )
    for script in (train, forecast):
        monkeypatch.setattr(script, "parse_args", lambda: args)
        monkeypatch.setattr(script, "DEQForecaster", TinyForecaster)
        monkeypatch.setattr(
            script, "build_loaders",
            lambda path, **kwargs: common.build_loaders(path, seq_len=2, pred_len=2, **kwargs),
        )
    train.main()
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["scaling"]["columns"] == ("OT", "aux")
    torch.testing.assert_close(payload["scaling"]["mean"], torch.tensor([2.5, 5.0]))
    args.data_path = forecast_path
    forecast.main()

    result = capsys.readouterr().out
    assert "first test window, variable OT" in result
    expected = round((10.0 - 2.5) / float(np.std(np.arange(6))), 3)
    assert f"actual   [{expected}" in result

    # Old weights-only checkpoints still evaluate using the supplied CSV.
    torch.save({"state_dict": payload["state_dict"], "sampler": "anderson"}, checkpoint)
    forecast.main()
    assert f"actual   [{expected}" not in capsys.readouterr().out


def test_saved_scaling_rejects_incompatible_columns(scripts, tmp_path):
    common, _, _ = scripts
    training = tmp_path / "original.csv"
    incomplete = tmp_path / "missing.csv"
    write_csv(training)
    write_csv(incomplete, missing=True)
    scaling = common.build_loaders(training, seq_len=2, pred_len=2)["train"].dataset.scaling
    with pytest.raises(ValueError, match="columns do not match"):
        common.build_loaders(incomplete, seq_len=2, pred_len=2, scaling=scaling)
