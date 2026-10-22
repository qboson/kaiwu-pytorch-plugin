"""build_loaders must reject look-backs or horizons larger than a split."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "qdeq" / "simple"
ROWS = 12 * 30 * 24 + 4 * 30 * 24 + 4 * 30 * 24


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


def test_a_look_back_longer_than_the_training_split_is_rejected(etth1):
    module, data_path = etth1

    with pytest.raises(ValueError, match="training split"):
        module.build_loaders(data_path, seq_len=9000, pred_len=96)


def test_a_horizon_longer_than_the_evaluation_split_is_rejected(etth1):
    module, data_path = etth1

    with pytest.raises(ValueError, match="evaluation split"):
        module.build_loaders(data_path, seq_len=96, pred_len=3000)


def test_the_largest_supported_windows_still_build(etth1):
    module, data_path = etth1

    loaders = module.build_loaders(data_path, seq_len=4000, pred_len=96)

    assert all(len(loaders[split].dataset) > 0 for split in ("train", "val", "test"))
