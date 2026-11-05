"""ETTh1Windows must reject window sizes and strides that cannot work."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "qdeq" / "simple"


def _module(monkeypatch):
    pandas = types.ModuleType("pandas")
    pandas.read_csv = lambda path: None
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    sys.modules.pop("etth1_common", None)
    import etth1_common

    return etth1_common


@pytest.mark.parametrize("kwargs", [{"seq_len": 0}, {"pred_len": 0}, {"seq_len": -6}])
def test_non_positive_window_sizes_are_rejected(monkeypatch, kwargs):
    module = _module(monkeypatch)
    values = np.arange(200, dtype=np.float32).reshape(200, 1)

    with pytest.raises(ValueError, match="positive"):
        module.ETTh1Windows(values, **{"seq_len": 10, "pred_len": 5, **kwargs})


@pytest.mark.parametrize("stride", [0, -2])
def test_non_positive_strides_are_rejected(monkeypatch, stride):
    module = _module(monkeypatch)
    values = np.arange(200, dtype=np.float32).reshape(200, 1)

    with pytest.raises(ValueError, match="stride"):
        module.ETTh1Windows(values, seq_len=10, pred_len=5, stride=stride)


def test_a_valid_window_still_builds(monkeypatch):
    module = _module(monkeypatch)
    values = np.arange(200, dtype=np.float32).reshape(200, 1)

    windows = module.ETTh1Windows(values, seq_len=10, pred_len=5, stride=2)

    assert len(windows) == len(range(0, 186, 2))
