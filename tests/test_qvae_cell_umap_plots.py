"""Contract test for ``example/qvae_cell/visualization.save_umap_plots``."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
VISUALIZATION = REPO / "example" / "qvae_cell" / "visualization.py"


def _load_visualization():
    scanpy = types.ModuleType("scanpy")
    scanpy.pl = types.SimpleNamespace(umap=lambda *args, **kwargs: None)
    previous = sys.modules.get("scanpy")
    sys.modules["scanpy"] = scanpy
    try:
        spec = importlib.util.spec_from_file_location(
            "qvae_cell_visualization_umap_uut", VISUALIZATION
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


def test_plots_labels_and_batches_on_separate_axes(tmp_path):
    visualization = _load_visualization()
    plt.close("all")
    calls: list[dict[str, object]] = []

    def fake_umap(adata, color=None, ax=None, show=None, title=None):
        calls.append(
            {"adata": adata, "color": color, "ax": ax, "show": show, "title": title}
        )
        ax.set_title(title)

    visualization.sc.pl.umap = fake_umap
    adata = object()

    visualization.save_umap_plots(adata, "cell_type", "batch", str(tmp_path))

    assert len(calls) == 2
    assert calls[0]["adata"] is adata
    assert calls[0]["color"] == "cell_type"
    assert calls[0]["title"] == "UMAP: Cell Types"
    assert calls[1]["color"] == "batch"
    assert calls[1]["title"] == "UMAP: Batches"
    assert calls[0]["show"] is False
    assert calls[1]["show"] is False
    # Both embeddings must land on different subplots.
    assert calls[0]["ax"] is not calls[1]["ax"]

    out_path = tmp_path / "qvae_umap_labels_batches.png"
    assert out_path.is_file()
    assert out_path.stat().st_size > 1000
    assert plt.get_fignums() == []


def test_creates_the_png_in_the_requested_directory(tmp_path):
    visualization = _load_visualization()
    visualization.sc.pl.umap = lambda adata, **kwargs: None

    nested = tmp_path / "nested"
    nested.mkdir()

    visualization.save_umap_plots(object(), "label", "batch", str(nested))

    assert sorted(path.name for path in nested.iterdir()) == [
        "qvae_umap_labels_batches.png"
    ]
