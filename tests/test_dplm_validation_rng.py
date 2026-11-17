"""Validation can reuse its input randomness without advancing the training stream."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

pytest.importorskip("tqdm")
pytest.importorskip("omegaconf")
pytest.importorskip("transformers")

from example.qdiffusion.dplm.utils import runtime
from example.qdiffusion.dplm.workflows.workflow_helpers import run_epoch


class StochasticObjective(nn.Module):
    def objective(self, batch):
        return {"energy_objective": torch.rand_like(batch["targets"])}


def test_seeded_validation_repeats_objective_and_restores_training_rng():
    model = StochasticObjective()
    loader = DataLoader([{"targets": torch.ones(3)} for _ in range(5)], batch_size=2)

    def run_validation():
        with runtime.validation_rng(71, "cpu"):
            return run_epoch(model, loader, optimizer=None, grad_clip_norm=0, description="val")

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(99)
        saved = torch.random.get_rng_state().clone()
        first = run_validation()
        torch.testing.assert_close(torch.random.get_rng_state(), saved)
        torch.rand(7)
        later = torch.random.get_rng_state().clone()
        second = run_validation()
        torch.testing.assert_close(torch.random.get_rng_state(), later)

    assert first == second


def test_validation_rng_restores_state_when_objective_fails():
    with torch.random.fork_rng(devices=[]):
        saved = torch.random.get_rng_state().clone()
        with pytest.raises(RuntimeError, match="objective failed"):
            with runtime.validation_rng(71, torch.device("cpu")):
                torch.rand(10)
                raise RuntimeError("objective failed")
        torch.testing.assert_close(torch.random.get_rng_state(), saved)
