"""Regression tests for the BM generation example Trainer cost weighting."""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch

pytest.importorskip("matplotlib")

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../example/bm_generation"))
)
import trainer as bm_trainer  # noqa: E402  pylint: disable=wrong-import-position


class _MatrixAwareSampler:
    """Minimal stand-in for a Kaiwu optimizer returning +-1 spin samples.

    The returned spins depend on the Ising matrix, so conditioning on
    different visible nodes produces different sampled states, as a real
    solver would.
    """

    def __init__(self, num_samples=4):
        self.num_samples = num_samples

    def solve(self, ising_mat):
        ising_mat = np.asarray(ising_mat, dtype=float)
        spins = np.where(ising_mat.sum(axis=1) >= 0, 1.0, -1.0)
        return np.tile(spins, (self.num_samples, 1))


class _RecordingSaver:
    """Saver that records the reported loss components."""

    def __init__(self):
        self.records = []

    def save_info(self, model, save_path, output_i, time):
        pass

    def output_loss(self, output_i, kl_div, ncl, func):
        self.records.append((kl_div, ncl, func))


class _InlinePool:
    """Runs ``map`` synchronously so the test does not fork worker processes."""

    def __init__(self, processes=1):
        self.processes = processes

    def map(self, func, iterable):
        return [func(item) for item in iterable]

    def close(self):
        pass

    def join(self):
        pass


def _run_one_step(beta, monkeypatch):
    monkeypatch.setattr(bm_trainer.mp, "Pool", _InlinePool)
    torch.manual_seed(0)
    saver = _RecordingSaver()
    trainer = bm_trainer.Trainer(
        data=[torch.rand(4, 6)],
        saver=saver,
        worker=_MatrixAwareSampler(),
        num_visible=6,
        num_hidden=4,
        num_output=2,
    )
    trainer.set_cost_parameter(alpha=0.25, beta=beta)
    with tempfile.TemporaryDirectory() as save_path:
        trainer.train(max_steps=1, save_path=save_path, num_processes=1)
    assert len(saver.records) == 1
    return saver.records[0]


def test_cost_parameter_beta_weights_ncl_term(monkeypatch):
    """The documented ``beta`` coefficient must weight the NCL term."""
    kl_short, ncl_short, cost_short = _run_one_step(beta=0.0, monkeypatch=monkeypatch)
    kl_long, ncl_long, cost_long = _run_one_step(beta=10.0, monkeypatch=monkeypatch)

    assert not np.isclose(ncl_long, 0.0), "NCL term is degenerate in this setup"
    assert np.isclose(cost_short, 0.25 * kl_short, atol=1e-6)
    assert np.isclose(cost_long, 0.25 * kl_long + 10.0 * ncl_long, atol=1e-6)
    assert not np.isclose(cost_short, cost_long)
