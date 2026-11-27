"""Checks that ETTh1 windows correspond to consecutive hours."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def example(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "example/qdeq/simple"
    monkeypatch.syspath_prepend(str(path))
    import etth1_common

    monkeypatch.setattr(etth1_common, "TRAIN_ROWS", 8)
    monkeypatch.setattr(etth1_common, "VAL_ROWS", 4)
    monkeypatch.setattr(etth1_common, "TEST_ROWS", 4)
    return etth1_common


@pytest.mark.parametrize("bad_date", ["duplicate", "gap", "reversed", "missing", "invalid"])
def test_bad_time_axis_is_rejected_before_windowing(example, tmp_path, bad_date):
    dates = list(pd.date_range("2020-01-01", periods=16, freq="h"))
    if bad_date == "duplicate":
        dates[8] = dates[7]
    elif bad_date == "gap":
        dates[8:] = [date + pd.Timedelta(hours=1) for date in dates[8:]]
    elif bad_date == "reversed":
        dates[7], dates[8] = dates[8], dates[7]
    elif bad_date == "missing":
        dates[8] = None
    else:
        dates[8] = "not a timestamp"
    csv = tmp_path / "ETTh1.csv"
    pd.DataFrame({"date": dates, "OT": np.arange(16)}).to_csv(csv, index=False)

    with pytest.raises(ValueError, match="consecutive hourly"):
        example.build_loaders(csv, seq_len=3, pred_len=2)


@pytest.mark.parametrize("numeric", [False, True])
def test_hourly_data_keeps_split_boundaries_and_training_scale(example, tmp_path, numeric):
    dates = np.arange(16) if numeric else pd.date_range("2020-01-01", periods=16, freq="h")
    values = np.arange(16, dtype=np.float32)
    csv = tmp_path / "ETTh1.csv"
    pd.DataFrame({"date": dates, "OT": values}).to_csv(csv, index=False)

    loaders = example.build_loaders(csv, seq_len=3, pred_len=2)
    scaled = (values - values[:8].mean()) / (values[:8].std() + 1e-8)
    for split, start in (("val", 8), ("test", 12)):
        history, target = loaders[split].dataset[0]
        torch.testing.assert_close(history[:, 0], torch.from_numpy(scaled[start - 3:start]))
        torch.testing.assert_close(target[:, 0], torch.from_numpy(scaled[start:start + 2]))
