"""Per-variable ETTh1 errors recover the units of the input CSV."""

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from torch import nn

pd = pytest.importorskip("pandas")


def test_original_unit_errors_use_the_same_windows_and_training_scale(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1] / "example" / "qdeq" / "simple"
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.setitem(sys.modules, "etth1_common", None)
    sys.modules.pop("etth1_common")
    common = importlib.import_module("etth1_common")
    for name, rows in (("TRAIN_ROWS", 4), ("VAL_ROWS", 2), ("TEST_ROWS", 5)):
        monkeypatch.setattr(common, name, rows)
    features = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
    values = np.arange(11)
    frame = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=11, freq="h"),
        **{name: 20 + slope * values for slope, name in enumerate(features, 1)},
    })
    path = tmp_path / "ETTh1.csv"
    frame.to_csv(path, index=False)
    loaders = common.build_loaders(path, seq_len=2, pred_len=2, batch_size=3, eval_stride=1)

    class LastValue(nn.Module):
        def __init__(self):
            super().__init__()
            self.deq = SimpleNamespace(get_last_stats=lambda: {"forward_nstep": 0})

        def forward(self, history):
            return history[:, -1:, :].expand(-1, 2, -1)

    test = loaders["test"]
    usual = common.evaluate(LastValue(), test)
    reported = common.evaluate(LastValue(), test, original_units=True)

    assert {key: reported[key] for key in usual} == usual
    assert list(reported["original_units"]) == list(features)
    for slope, name in enumerate(features, 1):
        assert reported["original_units"][name] == pytest.approx({
            "mse": 2.5 * slope ** 2, "mae": 1.5 * slope,
        })
