"""The ETTh1 loaders must scale every variable with its own statistics."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

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
    """Imports etth1_common with a stubbed pandas returning two very different scales."""
    rng = np.random.default_rng(0)
    values = np.empty((ROWS, 2), dtype=np.float32)
    values[:, 0] = rng.normal(0.0, 1.0, size=ROWS)
    values[:, 1] = rng.normal(1000.0, 500.0, size=ROWS)
    pandas = types.ModuleType("pandas")
    pandas.read_csv = lambda path: _Frame(values)
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    sys.modules.pop("etth1_common", None)
    import etth1_common

    data_path = tmp_path / "ETTh1.csv"
    data_path.write_text("date,HUFL,OT\n")
    yield etth1_common, data_path
    sys.modules.pop("etth1_common", None)


def test_each_column_is_standardized_with_its_own_statistics(etth1):
    module, data_path = etth1

    loaders = module.build_loaders(data_path)
    scaled = loaders["train"].dataset.values

    torch.testing.assert_close(scaled.mean(dim=0), torch.zeros(2), atol=1e-4, rtol=0)
    torch.testing.assert_close(scaled.std(dim=0, unbiased=False), torch.ones(2), atol=1e-4, rtol=0)
