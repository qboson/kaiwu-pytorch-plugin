"""Generated-sample visualisation contract for the DBN trainer."""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))

from dbn_trainer import DBNTrainer  # noqa: E402


class FakeRBM:
    """Returns twenty fixed samples and exposes the visible width."""

    num_visible = 4

    def sample(self, sampler):
        return torch.zeros(20, 8)


def test_twenty_samples_and_the_visible_slice_are_rendered():
    """Only the first twenty samples and the visible nodes reach the image."""
    recorded = {}

    def fake_image(data, size=8):
        recorded["shape"] = tuple(data.shape)
        return np.zeros((20, 20))

    trainer = DBNTrainer.__new__(DBNTrainer)
    trainer.sampler = object()
    trainer._gen_digits_image = fake_image
    trainer._visualize_generated_samples(FakeRBM(), batch_idx=2, epoch=6)
    figure = plt.gcf()
    try:
        assert recorded["shape"] == (20, 4)
        assert figure.axes[0].get_title() == "Generated Samples - Epoch 7, Batch 3"
    finally:
        plt.close("all")
