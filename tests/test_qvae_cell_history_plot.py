"""Contract tests for ``example/qvae_cell/visualization.plot_training_history``."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
VISUALIZATION = REPO / "example" / "qvae_cell" / "visualization.py"


def _load_visualization():
    # ``visualization`` imports scanpy at module import; the plotting helpers
    # under test do not need a real scanpy install.
    scanpy = types.ModuleType("scanpy")
    scanpy.pl = types.SimpleNamespace(umap=lambda *args, **kwargs: None)
    previous = sys.modules.get("scanpy")
    sys.modules["scanpy"] = scanpy
    try:
        spec = importlib.util.spec_from_file_location(
            "qvae_cell_visualization_uut", VISUALIZATION
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("scanpy", None)
        else:
            sys.modules["scanpy"] = previous


def _history(epochs: int = 3) -> dict[str, list[float]]:
    return {
        "train_loss": [1.0, 2.0, 3.0][:epochs],
        "val_loss": [4.0, 5.0, 6.0][:epochs],
        "train_recon_loss": [0.5, 0.6, 0.7][:epochs],
        "val_recon_loss": [0.8, 0.9, 1.0][:epochs],
        "train_kl": [0.1, 0.2, 0.3][:epochs],
        "val_kl": [0.4, 0.5, 0.6][:epochs],
    }


def test_writes_three_panels_and_closes_the_figure(tmp_path):
    visualization = _load_visualization()
    plt.close("all")
    out_path = tmp_path / "history.png"

    figure = visualization.plot_training_history(_history(), out_path=out_path)

    assert len(figure.axes) == 3
    # The third panel label is being reworded upstream, so pin the stable two.
    assert [axis.get_title() for axis in figure.axes][:2] == [
        "Total Loss",
        "Reconstruction Loss",
    ]
    assert figure.axes[2].get_title()
    for axis in figure.axes:
        lines = axis.get_lines()
        assert [line.get_label() for line in lines] == ["Train", "Val"]
        assert [line.get_linestyle() for line in lines] == ["-", "--"]
        assert axis.get_xlabel() == "Epoch"

    # The train series must stay on the first line and the validation series on
    # the second one, otherwise the two curves are silently swapped.
    assert list(figure.axes[0].get_lines()[0].get_ydata()) == [1.0, 2.0, 3.0]
    assert list(figure.axes[0].get_lines()[1].get_ydata()) == [4.0, 5.0, 6.0]
    assert figure.axes[0].get_xlim() == (1.0, 3.0)

    assert out_path.is_file()
    assert out_path.stat().st_size > 1000
    assert figure.number not in plt.get_fignums()


def test_single_epoch_uses_a_padded_x_range(tmp_path):
    visualization = _load_visualization()
    plt.close("all")
    out_path = tmp_path / "single.png"

    figure = visualization.plot_training_history(_history(epochs=1))

    # One epoch would otherwise collapse the x-axis to a degenerate range.
    assert figure.axes[0].get_xlim() == (0.5, 1.5)
    for axis in figure.axes:
        assert len(axis.get_lines()[0].get_xdata()) == 1
    assert not out_path.exists()
    plt.close(figure)


def test_requires_every_history_series():
    visualization = _load_visualization()
    plt.close("all")
    history = _history()
    del history["val_kl"]

    with pytest.raises(KeyError):
        visualization.plot_training_history(history)
    plt.close("all")
