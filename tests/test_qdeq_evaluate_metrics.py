"""evaluate must average per element and report the mean solver steps."""

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


class FakeDeq:
    """Returns one solver-step count per forward."""

    def __init__(self, steps):
        self._steps = list(steps)

    def get_last_stats(self):
        """Pops the next step count."""
        return {"forward_nstep": self._steps.pop(0)}


class FakeModel:
    """Returns the prepared forecasts and step counts."""

    def __init__(self, outputs, steps):
        self._outputs = list(outputs)
        self.deq = FakeDeq(steps)

    def eval(self):
        """Marks evaluation mode."""
        return self

    def __call__(self, history):
        """Returns the next prepared forecast."""
        del history
        return self._outputs.pop(0)


def test_metrics_average_per_element_and_report_mean_steps(monkeypatch):
    module = _module(monkeypatch)
    targets = torch.cat([torch.zeros(1, 3, 2), torch.full((3, 3, 2), 2.0)])
    dataset = TensorDataset(torch.zeros_like(targets), targets)
    loader = DataLoader(dataset, batch_sampler=[[0], [1, 2, 3]])
    model = FakeModel(
        outputs=[torch.ones(1, 3, 2), torch.full((3, 3, 2), 4.0)],
        steps=[3, 5],
    )

    metrics = module.evaluate(model, loader)

    assert metrics["mse"] == 3.25
    assert metrics["mae"] == 1.75
    assert metrics["steps"] == 4.0
