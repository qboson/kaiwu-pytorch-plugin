"""ETTh1Windows must cover every complete window without overlap or truncation."""

import sys
import types
from pathlib import Path

import numpy as np
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "qdeq" / "simple"


def _windows_module(monkeypatch):
    pandas = types.ModuleType("pandas")
    pandas.read_csv = lambda path: None
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    sys.modules.pop("etth1_common", None)
    import etth1_common

    return etth1_common


def test_every_complete_window_is_returned_in_order(monkeypatch):
    module = _windows_module(monkeypatch)
    values = np.arange(1000, dtype=np.float32).reshape(1000, 1)

    windows = module.ETTh1Windows(values, seq_len=10, pred_len=5, stride=1)

    assert len(windows) == 1000 - 10 - 5 + 1
    assert windows.starts == list(range(986))
    history, target = windows[0]
    assert history[0, 0].item() == 0.0
    assert target[-1, 0].item() == 14.0
    history, target = windows[-1]
    assert history[0, 0].item() == 985.0
    assert target[-1, 0].item() == 999.0


def test_history_and_target_are_contiguous(monkeypatch):
    module = _windows_module(monkeypatch)
    values = np.arange(100, dtype=np.float32).reshape(100, 1)

    windows = module.ETTh1Windows(values, seq_len=7, pred_len=3, stride=1)

    history, target = windows[3]
    assert history.shape == (7, 1)
    assert target.shape == (3, 1)
    assert target[0, 0].item() == history[-1, 0].item() + 1.0


def test_stride_subsamples_the_starts(monkeypatch):
    module = _windows_module(monkeypatch)
    values = np.arange(100, dtype=np.float32).reshape(100, 1)

    windows = module.ETTh1Windows(values, seq_len=10, pred_len=5, stride=2)

    assert windows.starts == list(range(0, 86, 2))
    history, target = windows[-1]
    assert history[0, 0].item() == 84.0
    assert target[-1, 0].item() == 98.0
    assert isinstance(history, torch.Tensor)
