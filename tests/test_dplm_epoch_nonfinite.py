"""Reject invalid DPLM training objectives before updating parameters."""

import pytest
import torch
from torch import nn

pytest.importorskip("omegaconf")
pytest.importorskip("transformers")

from example.qdiffusion.dplm.workflows.workflow_helpers import run_epoch


class ScalarObjective(nn.Module):
    """Small differentiable objective for exercising the actual epoch loop."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def objective(self, batch):
        return {"energy_objective": self.weight * batch["targets"]}


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
@pytest.mark.parametrize("training", [False, True])
def test_epoch_rejects_nonfinite_objective(nonfinite, training):
    generator = ScalarObjective()
    optimizer = torch.optim.AdamW(generator.parameters()) if training else None

    with pytest.raises(FloatingPointError, match=r"example-epoch.*batch 1.*non-finite"):
        run_epoch(
            generator, [{"targets": torch.tensor([nonfinite])}],
            optimizer=optimizer, grad_clip_norm=0, description="example-epoch",
        )

    torch.testing.assert_close(generator.weight, torch.ones(()))
    assert generator.weight.grad is None
    if optimizer is not None:
        assert not optimizer.state


def test_epoch_keeps_sample_weighted_finite_validation_metric():
    result = run_epoch(
        ScalarObjective(), [{"targets": torch.tensor([1., 3.])}, {"targets": torch.tensor([8.])}],
        optimizer=None, grad_clip_norm=0, description="validation",
    )
    assert result["energy_objective_mean"] == pytest.approx(4.0)
