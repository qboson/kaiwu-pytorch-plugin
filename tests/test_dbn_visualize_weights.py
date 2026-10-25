"""Weight/gradient visualisation contract for the DBN trainer."""

import os
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))

from dbn_trainer import DBNTrainer  # noqa: E402


def build_rbm():
    """Two visible and three hidden units with known gradients."""
    coef = torch.arange(6, dtype=torch.float32, requires_grad=True).reshape(2, 3)
    coef.grad = torch.ones(2, 3)
    return SimpleNamespace(
        quadratic_coef=coef,
        linear_bias=torch.arange(5, dtype=torch.float32),
        num_visible=2,
    )


def test_three_panels_render_weights_gradients_and_hidden_bias():
    """The figure carries the three panels and plots the hidden bias heights."""
    trainer = DBNTrainer.__new__(DBNTrainer)
    trainer._visualize_weights_gradients(build_rbm(), batch_idx=0, epoch=0)
    figure = plt.gcf()
    try:
        titles = [axis.get_title() for axis in figure.axes]
        assert "Weight Matrix" in titles
        assert "Weight Gradients" in titles
        assert "Hidden Unit Biases" in titles
        bars = [axis for axis in figure.axes if axis.patches]
        assert len(bars) == 1
        assert [patch.get_height() for patch in bars[0].patches] == [2.0, 3.0, 4.0]
    finally:
        plt.close("all")
