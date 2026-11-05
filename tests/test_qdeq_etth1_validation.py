"""build_loaders must reject data too short for its fixed split and bad strides."""

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
    """Imports etth1_common with a stubbed pandas and a fixed row count."""
    state = {"rows": ROWS}
    pandas = types.ModuleType("pandas")
    pandas.read_csv = lambda path: _Frame(np.zeros((state["rows"], 7), dtype=np.float32))
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    sys.modules.pop("etth1_common", None)
    import etth1_common

    data_path = tmp_path / "ETTh1.csv"
    data_path.write_text("date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT\n")
    yield etth1_common, state, data_path
    sys.modules.pop("etth1_common", None)


def test_a_short_csv_is_rejected(etth1):
    module, state, data_path = etth1
    state["rows"] = 5000

    with pytest.raises(ValueError, match="at least"):
        module.build_loaders(data_path)


def test_zero_sampling_stride_is_rejected(etth1):
    module, _, data_path = etth1

    with pytest.raises(ValueError, match="stride"):
        module.build_loaders(data_path, train_stride=0)


def test_negative_evaluation_stride_is_rejected(etth1):
    module, _, data_path = etth1

    with pytest.raises(ValueError, match="stride"):
        module.build_loaders(data_path, eval_stride=-4)


def test_a_full_length_csv_still_builds_three_loaders(etth1):
    module, _, data_path = etth1

    loaders = module.build_loaders(data_path)

    assert set(loaders) == {"train", "val", "test"}
    for split, loader in loaders.items():
        history, target = next(iter(loader))
        assert history.shape[0] == target.shape[0] > 0, split
        assert history.shape[2] == target.shape[2] == 7, split
