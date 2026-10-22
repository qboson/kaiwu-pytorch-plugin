"""The three ETTh1 loaders must use their documented strides and batch size."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "qdeq" / "simple"
TRAIN_ROWS, VAL_ROWS, TEST_ROWS = 12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24
ROWS = TRAIN_ROWS + VAL_ROWS + TEST_ROWS
SEQ_LEN = PRED_LEN = 96


class _Frame:
    """Stand-in for the DataFrame returned by pandas.read_csv."""

    def __init__(self, values):
        self._values = values

    def drop(self, columns):
        """Ignores the dropped date column like the real frame."""
        assert columns == ["date"]
        return self

    def to_numpy(self, dtype=None):
        """Returns the stored values in the requested dtype."""
        return self._values.astype(dtype)


@pytest.fixture
def etth1(monkeypatch, tmp_path):
    """Imports etth1_common with a stubbed pandas and a full-length series."""
    pandas = types.ModuleType("pandas")
    pandas.read_csv = lambda path: _Frame(np.zeros((ROWS, 7), dtype=np.float32))
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    sys.modules.pop("etth1_common", None)
    import etth1_common

    data_path = tmp_path / "ETTh1.csv"
    data_path.write_text("date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT\n")
    yield etth1_common, data_path
    sys.modules.pop("etth1_common", None)


def _windows(rows, stride):
    return len(range(0, rows - SEQ_LEN - PRED_LEN + 1, stride))


def test_each_split_uses_its_documented_stride(etth1):
    module, data_path = etth1

    loaders = module.build_loaders(data_path)

    assert len(loaders["train"].dataset) == _windows(TRAIN_ROWS, 4)
    assert len(loaders["val"].dataset) == _windows(VAL_ROWS + SEQ_LEN, 8)
    assert len(loaders["test"].dataset) == _windows(TEST_ROWS + SEQ_LEN, 8)
    assert len(loaders["train"]) == 67
    assert len(loaders["val"]) == 11
    assert len(loaders["test"]) == 11


def test_batches_keep_the_configured_batch_size(etth1):
    module, data_path = etth1

    loaders = module.build_loaders(data_path)

    for split in ("train", "val", "test"):
        history, target = next(iter(loaders[split]))
        assert history.shape == (32, SEQ_LEN, 7), split
        assert target.shape == (32, PRED_LEN, 7), split
