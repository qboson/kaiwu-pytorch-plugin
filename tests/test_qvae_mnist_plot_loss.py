"""Loss-curve plot contract for the QVAE-MNIST helpers."""

import importlib.util
import os
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "qvae_mnist_helpers_plot_loss",
    os.path.join(REPO_ROOT, "example", "qvae_mnist", "utils", "helpers.py"),
)
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)


def test_title_reports_the_final_loss_of_one_curve():
    """The figure title carries the last loss with three decimals."""
    history = SimpleNamespace(history={"loss": [1.0, 0.5]}, epoch=[0, 1])
    helpers.plot_loss(history)
    figure = plt.gcf()
    try:
        assert len(figure.axes) == 1
        assert figure.axes[0].get_title() == "Loss: 0.500"
    finally:
        plt.close(figure)
