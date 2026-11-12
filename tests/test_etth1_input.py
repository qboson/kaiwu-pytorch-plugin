"""Data integrity checks for the QDEQ forecasting example."""

import csv

import numpy as np
import pytest
import torch

pytest.importorskip("pandas")

from example.qdeq.simple import etth1_common


@pytest.fixture
def etth1_csv(tmp_path, monkeypatch):
    """Use short chronological splits with the same boundary logic as ETTh1."""
    monkeypatch.setattr(etth1_common, "TRAIN_ROWS", 6)
    monkeypatch.setattr(etth1_common, "VAL_ROWS", 3)
    monkeypatch.setattr(etth1_common, "TEST_ROWS", 3)

    def write(values):
        path = tmp_path / "电力.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "OT"])
            writer.writerows(enumerate(values))
        return path

    return write


def test_rejects_truncated_test_split(etth1_csv):
    path = etth1_csv(range(11))  # Still enough rows to form test windows.
    with pytest.raises(ValueError, match=r"12 rows.*11"):
        etth1_common.build_loaders(path, seq_len=2, pred_len=1)


@pytest.mark.parametrize("bad_value", ["", "inf", "-inf"])
def test_rejects_nonfinite_observations_in_test_split(etth1_csv, bad_value):
    path = etth1_csv([*range(11), bad_value])
    with pytest.raises(ValueError, match=r"non-finite.*OT.*12"):
        etth1_common.build_loaders(path, seq_len=2, pred_len=1)


def test_valid_test_windows_use_training_scale_and_stop_at_split_end(etth1_csv):
    path = etth1_csv([*range(12), "not used"])
    loaders = etth1_common.build_loaders(path, seq_len=2, pred_len=1, eval_stride=1)
    test = loaders["test"].dataset
    history, target = test[0]
    train_mean, train_std = np.mean(np.arange(6)), np.std(np.arange(6))

    assert len(test) == 3
    torch.testing.assert_close(history[:, 0], (torch.tensor([7., 8.]) - train_mean) / train_std)
    torch.testing.assert_close(target[:, 0], (torch.tensor([9.]) - train_mean) / train_std)
