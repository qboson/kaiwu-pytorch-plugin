"""Sampling-worker contract for the BM generation example."""

import importlib.util
import os

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "bm_generation_trainer_worker",
    os.path.join(REPO_ROOT, "example", "bm_generation", "trainer.py"),
)
trainer_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trainer_module)


class FakeBM:
    """Records condition_sample calls and doubles the visible input."""

    def __init__(self):
        self.calls = []

    def condition_sample(self, sampler, s_visible):
        self.calls.append((sampler, tuple(s_visible.shape)))
        return s_visible * 2


def test_worker_samples_the_full_and_conditioned_chunks():
    """Two condition_sample calls are made: the full chunk and one without outputs."""
    bm = FakeBM()
    sampler = object()
    chunk = torch.arange(6, dtype=torch.float32).view(2, 3).requires_grad_(True)
    state_v, state_vi = trainer_module.process_solve_graph_worker(
        (bm, sampler, chunk, 1)
    )
    assert bm.calls == [(sampler, (2, 3)), (sampler, (2, 2))]
    assert state_v.tolist() == [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]]
    assert state_vi.tolist() == [[0.0, 2.0], [6.0, 8.0]]
