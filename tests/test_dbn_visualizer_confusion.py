"""Contract test for the DBN visualiser confusion-matrix artifact."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import matplotlib
import numpy as np


REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "example" / "dbn_digits"
VISUALIZER = EXAMPLE / "supervised_dbn_digits.py"


def _load_visualizer():
    matplotlib.use("Agg")
    # The plotting target does not use DBN training. Do not import its SDK path.
    missing = object()
    previous = sys.modules.get("dbn_trainer", missing)
    trainer = types.ModuleType("dbn_trainer")

    def _unused_pretrainer(*_args, **_kwargs):
        raise AssertionError("DBN training is outside this plotting fixture")

    trainer.DBNPretrainer = _unused_pretrainer
    sys.modules["dbn_trainer"] = trainer
    sys.path.insert(0, str(EXAMPLE))
    try:
        spec = importlib.util.spec_from_file_location(
            "dbn_digits_visualizer_uut", VISUALIZER
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE))
        if previous is missing:
            sys.modules.pop("dbn_trainer", None)
        else:
            sys.modules["dbn_trainer"] = previous


def _capture_matrices(module, monkeypatch):
    matrices = []
    draw_heatmap = module.sns.heatmap

    def record_and_draw(matrix, *args, **kwargs):
        """Capture the true matrix before drawing the temporary plot."""
        matrices.append(np.asarray(matrix).copy())
        return draw_heatmap(matrix, *args, **kwargs)

    monkeypatch.setattr(module.sns, "heatmap", record_and_draw)
    monkeypatch.setattr(module.plt, "show", lambda: None)
    return matrices


def test_writes_the_pdf_with_the_requested_suffix(tmp_path, monkeypatch):
    """Preserve the true confusion counts and requested PDF suffix."""
    module = _load_visualizer()
    plt = module.plt
    matrices = _capture_matrices(module, monkeypatch)
    plt.close("all")
    visualizer = module.RBMVisualizer(result_dir=str(tmp_path))

    visualizer.plot_confusion_matrix(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        title_suffix="fold1",
        save_pdf=True,
    )

    output = tmp_path / "confusion_matrix_fold1.pdf"
    assert output.is_file()
    assert output.stat().st_size > 1000
    assert len(matrices) == 1
    np.testing.assert_array_equal(matrices[0], [[1, 1], [0, 2]])
    plt.close("all")


def test_skips_the_file_when_pdf_output_is_disabled(tmp_path, monkeypatch):
    """Compute the true confusion counts without writing any PDF."""
    module = _load_visualizer()
    plt = module.plt
    matrices = _capture_matrices(module, monkeypatch)
    plt.close("all")
    visualizer = module.RBMVisualizer(result_dir=str(tmp_path))

    visualizer.plot_confusion_matrix(
        [0, 1],
        [0, 1],
        title_suffix="nopdf",
        save_pdf=False,
    )

    assert not (tmp_path / "confusion_matrix_nopdf.pdf").exists()
    assert not list(tmp_path.glob("*.pdf"))
    assert len(matrices) == 1
    np.testing.assert_array_equal(matrices[0], [[1, 0], [0, 1]])
    plt.close("all")
