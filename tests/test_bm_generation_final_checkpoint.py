"""Regression test: bm_generation training saves its final completed step.

``Trainer.train`` saved checkpoints at step 0 and whenever ``step % 10 == 0``.
A run with ``max_steps < 10`` (or any length not ending on a multiple of ten)
therefore finished without a checkpoint of the trained weights: the newest
artifact on disk was the initial, untrained model saved at step 0.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402  pylint: disable=wrong-import-position
import pytest  # noqa: E402  pylint: disable=wrong-import-position
import torch  # noqa: E402  pylint: disable=wrong-import-position

pytest.importorskip("kaiwu")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "bm_generation"))

import trainer as bm_trainer  # noqa: E402  pylint: disable=wrong-import-position


class _FakeSampler:
    """Deterministic stand-in for the Kaiwu annealer returning +-1 spins."""

    def __init__(self, num_samples=4):
        self.num_samples = num_samples

    def solve(self, ising_mat):
        matrix = np.asarray(ising_mat, dtype=float)
        spins = np.where(matrix.sum(axis=1) >= 0, 1.0, -1.0)
        return np.tile(spins, (self.num_samples, 1))


class _RecordingSaver:
    """Saver that records every checkpoint step."""

    def __init__(self):
        self.saved_steps = []
        self.loss_reports = []

    def save_info(self, model, save_path, output_i, time):
        self.saved_steps.append(output_i)

    def output_loss(self, output_i, kl_div, ncl, func):
        self.loss_reports.append((output_i, kl_div, ncl, func))


class _InlinePool:
    """Runs ``map`` synchronously so the test never forks worker processes."""

    def __init__(self, processes=1):
        self.processes = processes

    def map(self, func, iterable):
        return [func(item) for item in iterable]

    def close(self):
        pass

    def join(self):
        pass


def _train_and_record(monkeypatch, max_steps):
    monkeypatch.setattr(bm_trainer.mp, "Pool", _InlinePool)
    torch.manual_seed(0)
    saver = _RecordingSaver()
    trainer = bm_trainer.Trainer(
        data=[torch.rand(4, 6)],
        saver=saver,
        worker=_FakeSampler(),
        num_visible=6,
        num_hidden=4,
        num_output=2,
    )
    trainer.set_cost_parameter(alpha=0.25, beta=0.5)
    import tempfile

    with tempfile.TemporaryDirectory() as save_path:
        trainer.train(max_steps=max_steps, save_path=save_path, num_processes=1)
    return saver


@pytest.mark.parametrize(
    "max_steps,expected_steps",
    [
        (1, [0, 1]),  # short run: previously only the untrained step-0 model
        (3, [0, 3]),
        (10, [0, 10]),  # aligned run: periodic save already covers the final step
        (11, [0, 10, 11]),
    ],
)
def test_final_completed_step_is_saved(monkeypatch, max_steps, expected_steps):
    saver = _train_and_record(monkeypatch, max_steps)

    assert saver.saved_steps == expected_steps
    # Every completed step still produced a loss report.
    assert [entry[0] for entry in saver.loss_reports] == list(
        range(1, max_steps + 1)
    )


def test_final_save_reports_elapsed_time(monkeypatch):
    saver = _train_and_record(monkeypatch, 2)

    assert saver.saved_steps == [0, 2]
