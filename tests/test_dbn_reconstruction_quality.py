"""Reconstruction-quality logging contract for the DBN trainer."""

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import dbn_trainer as trainer_module  # noqa: E402


def test_evaluation_is_capped_at_one_hundred_samples(monkeypatch, capsys):
    """At most 100 rows are reconstructed and the mean error is printed."""
    recorded = {}

    def fake_reconstruct(rbm, data, device=None):
        recorded["shape"] = tuple(data.shape)
        return data, np.full(data.shape[0], 0.5)

    monkeypatch.setattr(
        trainer_module.UnsupervisedDBN, "reconstruct_with_rbm", fake_reconstruct
    )
    trainer = trainer_module.DBNTrainer.__new__(trainer_module.DBNTrainer)
    data = torch.zeros(150, 4)
    trainer._evaluate_reconstruction_quality("rbm", data, epoch=0, layer_idx=1)

    assert recorded["shape"] == (100, 4)
    output = capsys.readouterr().out
    assert "[RBM] Layer 2, Epoch 1" in output
    assert "Reconstruction Error = 0.500000" in output
