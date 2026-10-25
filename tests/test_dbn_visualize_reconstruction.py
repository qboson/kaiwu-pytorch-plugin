"""Reconstruction visualisation contract for the DBN trainer."""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))

import dbn_trainer as trainer_module  # noqa: E402
from dbn_trainer import DBNTrainer  # noqa: E402


def test_one_row_per_sample_is_rendered(tmp_path, monkeypatch):
    """Eight original/reconstruction pairs are drawn for a two-row batch."""
    calls = {}

    def fake_reconstruct(rbm, data, device=None):
        calls["shape"] = tuple(data.shape)
        return np.zeros((data.shape[0], 64)), np.zeros(data.shape[0])

    monkeypatch.setattr(
        trainer_module.UnsupervisedDBN, "reconstruct_with_rbm", fake_reconstruct
    )
    trainer = DBNTrainer.__new__(DBNTrainer)
    batch = torch.zeros(2, 64)
    trainer._visualize_current_reconstruction("rbm", batch, batch_idx=0, epoch=0)
    figure = plt.gcf()
    try:
        assert calls["shape"] == (2, 64)
        titles = [axis.get_title() for axis in figure.axes]
        assert "Original 1" in titles and "Original 2" in titles
        assert "Reconstructed 1" in titles and "Reconstructed 2" in titles
    finally:
        plt.close("all")
