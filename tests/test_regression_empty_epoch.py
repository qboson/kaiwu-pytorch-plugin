"""An empty loader must not fabricate a perfect validation objective."""
import importlib
from pathlib import Path
import pytest
import torch
from torch import nn


class Objective(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))

    def objective(self, batch):
        return {"energy_objective": self.scale * batch["targets"].float().sum(-1)}


@pytest.fixture
def run_epoch(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("example.qdiffusion.dplm.workflows.workflow_helpers").run_epoch


@pytest.mark.parametrize("training", [False, True])
def test_empty_epoch_raises_before_reporting_a_loss(run_epoch, training):
    model = Objective()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01) if training else None
    with pytest.raises(ValueError, match="empty|at least one"):
        run_epoch(model, [], optimizer=optimizer, grad_clip_norm=0., description="empty")
    assert model.scale.item() == 1.


def test_zero_size_batches_are_not_observations(run_epoch):
    with pytest.raises(ValueError, match="empty|at least one"):
        run_epoch(Objective(), [{"targets": torch.empty(0, 2)}], optimizer=None,
                  grad_clip_norm=0., description="zero")


def test_nonempty_epoch_retains_sample_weighting(run_epoch):
    loader = [{"targets": torch.tensor([[1., 1.], [2., 2.]])},
              {"targets": torch.tensor([[5., 5.]])}]
    result = run_epoch(Objective(), loader, optimizer=None, grad_clip_norm=0., description="valid")
    assert result["energy_objective_mean"] == pytest.approx(16. / 3)
