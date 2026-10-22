"""The ETTh1 loaders must align splits at the boundaries and scale with train data."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "qdeq" / "simple"
TRAIN_ROWS, VAL_ROWS, TEST_ROWS = 12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24
ROWS = TRAIN_ROWS + VAL_ROWS + TEST_ROWS


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
    """Imports etth1_common with a stubbed pandas and configurable values."""
    state = {"values": np.tile(np.arange(ROWS, dtype=np.float32)[:, None], (1, 7))}
    pandas = types.ModuleType("pandas")
    pandas.read_csv = lambda path: _Frame(state["values"])
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    sys.modules.pop("etth1_common", None)
    import etth1_common

    data_path = tmp_path / "ETTh1.csv"
    data_path.write_text("date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT\n")
    yield etth1_common, state, data_path
    sys.modules.pop("etth1_common", None)


def _scaled(values, index):
    train = values[:TRAIN_ROWS]
    mean, std = train.mean(axis=0), train.std(axis=0) + 1e-8
    return (values[index] - mean) / std


def test_validation_and_test_start_at_their_split_boundaries(etth1):
    module, state, data_path = etth1
    values = state["values"]
    loaders = module.build_loaders(data_path)

    val_history, val_target = next(iter(loaders["val"]))
    test_history, test_target = next(iter(loaders["test"]))

    np.testing.assert_allclose(val_history[0, -1], _scaled(values, TRAIN_ROWS - 1), rtol=1e-5)
    np.testing.assert_allclose(val_target[0, 0], _scaled(values, TRAIN_ROWS), rtol=1e-5)
    np.testing.assert_allclose(
        test_history[0, -1], _scaled(values, TRAIN_ROWS + VAL_ROWS - 1), rtol=1e-5
    )
    np.testing.assert_allclose(
        test_target[0, 0], _scaled(values, TRAIN_ROWS + VAL_ROWS), rtol=1e-5
    )


def test_scaling_uses_training_statistics_only(etth1):
    module, state, data_path = etth1
    values = state["values"].copy()
    values[TRAIN_ROWS:] += 1000.0
    state["values"] = values

    loaders = module.build_loaders(data_path)
    _, val_target = next(iter(loaders["val"]))

    np.testing.assert_allclose(val_target[0, 0], _scaled(values, TRAIN_ROWS), rtol=1e-5)
