"""Regression test for the training iteration demonstrated by example/run_rbm.py."""

import os
import runpy
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("kaiwu")

import kaiwu.classical as kaiwu_classical  # noqa: E402

EXAMPLE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../example/run_rbm.py")
)


class _StubSampler:
    def solve(self, ising_mat):
        num_spins = ising_mat.shape[0] - 1
        rng = np.random.default_rng(0)
        return rng.choice([-1.0, 1.0], size=(1, num_spins + 1))


class _RecordingSGD(torch.optim.SGD):
    """SGD that records whether step() ran and how the parameters changed."""

    instances = []

    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.step_called = False
        self.before = [p.detach().clone() for p in self.param_groups[0]["params"]]
        self.after = None
        _RecordingSGD.instances.append(self)

    def step(self, *args, **kwargs):
        self.step_called = True
        result = super().step(*args, **kwargs)
        self.after = [p.detach().clone() for p in self.param_groups[0]["params"]]
        return result


def test_run_rbm_updates_the_model_weights(monkeypatch):
    monkeypatch.setattr(kaiwu_classical, "SimulatedAnnealingOptimizer", _StubSampler)
    monkeypatch.setattr(torch.optim, "SGD", _RecordingSGD)
    _RecordingSGD.instances = []

    runpy.run_path(EXAMPLE, run_name="__main__")

    assert len(_RecordingSGD.instances) == 1
    optimizer = _RecordingSGD.instances[0]
    assert optimizer.step_called, "the example never updates the model weights"
    assert any(
        not torch.equal(before, after)
        for before, after in zip(optimizer.before, optimizer.after)
    )
