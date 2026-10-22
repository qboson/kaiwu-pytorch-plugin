"""evaluate must switch the model to eval mode and run without gradients."""

import sys
import types
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "qdeq" / "simple"


def _module(monkeypatch):
    pandas = types.ModuleType("pandas")
    pandas.read_csv = lambda path: None
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    sys.modules.pop("etth1_common", None)
    import etth1_common

    return etth1_common


class RecordingModel:
    """Records eval()-mode and grad-mode for every forward."""

    def __init__(self):
        self.eval_called = False
        self.grad_enabled = []
        self.deq = types.SimpleNamespace(get_last_stats=lambda: {"forward_nstep": 1})

    def eval(self):
        """Records that evaluate switched the model to eval mode."""
        self.eval_called = True
        return self

    def __call__(self, history):
        """Records the grad mode and returns the history."""
        self.grad_enabled.append(torch.is_grad_enabled())
        return history


def test_evaluate_switches_to_eval_mode(monkeypatch):
    module = _module(monkeypatch)
    targets = torch.zeros(4, 96, 3)
    loader = DataLoader(TensorDataset(targets, targets), batch_size=2)
    model = RecordingModel()

    module.evaluate(model, loader)

    assert model.eval_called is True


def test_evaluate_runs_without_gradients(monkeypatch):
    module = _module(monkeypatch)
    targets = torch.zeros(4, 96, 3)
    loader = DataLoader(TensorDataset(targets, targets), batch_size=2)
    model = RecordingModel()

    module.evaluate(model, loader)

    assert model.grad_enabled == [False, False]
