"""ETTh1Windows must hand float32 tensors to the float32 model."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "qdeq" / "simple"


def _module(monkeypatch):
    pandas = types.ModuleType("pandas")
    pandas.read_csv = lambda path: None
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    sys.modules.pop("etth1_common", None)
    import etth1_common

    return etth1_common


def test_float64_input_is_converted_to_float32(monkeypatch):
    module = _module(monkeypatch)
    values = np.arange(400, dtype=np.float64).reshape(200, 2)

    windows = module.ETTh1Windows(values, seq_len=10, pred_len=5)

    history, target = windows[0]
    assert windows.values.dtype == torch.float32
    assert history.dtype == torch.float32
    assert target.dtype == torch.float32
    assert history.shape == (10, 2)
    assert target.shape == (5, 2)
