"""History-comparison plot contract for the QVAE-MNIST helpers."""

import importlib.util
import os
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "qvae_mnist_helpers_history",
    os.path.join(REPO_ROOT, "example", "qvae_mnist", "utils", "helpers.py"),
)
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)


def history(losses, accs):
    """Builds the Keras-like history object the helper expects."""
    return SimpleNamespace(
        history={"loss": losses, "acc": accs, "val_loss": losses},
        epoch=list(range(len(losses))),
    )


def test_accuracy_panel_is_added_when_requested():
    """Two models produce a loss panel and an accuracy panel."""
    histories = [history([1.0, 0.5], [0.6, 0.8]), history([2.0, 1.0], [0.5, 0.7])]
    helpers.plot_compare_histories(histories, ["a", "b"], plot_accuracy=True)
    figure = plt.gcf()
    axes = figure.axes
    try:
        assert len(axes) == 2
        assert axes[0].get_title().startswith("Training Loss:")
        assert axes[1].get_title() == "Accuracy"
    finally:
        plt.close(figure)


def test_loss_only_panel_can_be_requested():
    """``plot_accuracy=False`` keeps the single loss panel."""
    histories = [history([1.0, 0.5], [0.6, 0.8])]
    helpers.plot_compare_histories(histories, ["a"], plot_accuracy=False)
    figure = plt.gcf()
    try:
        assert len(figure.axes) == 1
        assert figure.axes[0].get_title().startswith("Training Loss:")
    finally:
        plt.close(figure)
