"""Only the training loader may shuffle its windows."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import RandomSampler, SequentialSampler

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


def test_only_the_training_split_is_shuffled(etth1):
    module, data_path = etth1

    loaders = module.build_loaders(data_path)

    assert isinstance(loaders["train"].sampler, RandomSampler)
    assert isinstance(loaders["val"].sampler, SequentialSampler)
    assert isinstance(loaders["test"].sampler, SequentialSampler)
