"""Contract test for ``example/qvae_cell/visualization.save_energy_plots``."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
VISUALIZATION = REPO / "example" / "qvae_cell" / "visualization.py"


def _load_visualization():
    scanpy = types.ModuleType("scanpy")
    scanpy.pl = types.SimpleNamespace(umap=lambda *args, **kwargs: None)
    previous = sys.modules.get("scanpy")
    sys.modules["scanpy"] = scanpy
    try:
        spec = importlib.util.spec_from_file_location(
            "qvae_cell_visualization_energy_uut", VISUALIZATION
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


def _adata():
    energy = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    labels = ["B"] * 5 + ["A"] * 5
    return types.SimpleNamespace(obs=pd.DataFrame({"QVAE_Energy": energy, "cell_type": labels}))


def test_energy_umap_uses_five_to_ninety_five_percentiles(tmp_path):
    visualization = _load_visualization()
    plt.close("all")
    umap_calls: list[dict[str, object]] = []
    violin_calls: list[dict[str, object]] = []

    def fake_umap(adata, **kwargs):
        umap_calls.append(kwargs)
        plt.figure()

    def fake_violin(**kwargs):
        violin_calls.append(kwargs)

    visualization.sc.pl.umap = fake_umap
    visualization.sns.violinplot = fake_violin
    adata = _adata()

    visualization.save_energy_plots(adata, "cell_type", str(tmp_path))

    energy = adata.obs["QVAE_Energy"]
    assert len(umap_calls) == 1
    assert umap_calls[0]["color"] == "QVAE_Energy"
    assert umap_calls[0]["title"] == "UMAP: QVAE Energy"
    assert umap_calls[0]["show"] is False
    assert umap_calls[0]["vmin"] == np.percentile(energy, 5)
    assert umap_calls[0]["vmax"] == np.percentile(energy, 95)
    assert umap_calls[0]["cmap"] == "coolwarm"

    assert len(violin_calls) == 1
    assert violin_calls[0]["x"] == "cell_type"
    assert violin_calls[0]["y"] == "QVAE_Energy"
    assert violin_calls[0]["data"] is adata.obs
    assert violin_calls[0]["inner"] == "quartile"
    # Cell types are ordered by their median energy: B is the low group here.
    assert list(violin_calls[0]["order"]) == ["B", "A"]

    assert (tmp_path / "qvae_energy_umap.png").is_file()
    assert (tmp_path / "qvae_energy_by_celltype.png").is_file()
    assert plt.get_fignums() == []


def test_writes_both_energy_figures_into_the_output_directory(tmp_path):
    visualization = _load_visualization()
    visualization.sc.pl.umap = lambda adata, **kwargs: plt.figure()
    visualization.sns.violinplot = lambda **kwargs: None

    out_dir = tmp_path / "figures"
    out_dir.mkdir()

    visualization.save_energy_plots(_adata(), "cell_type", str(out_dir))

    assert sorted(path.name for path in out_dir.iterdir()) == [
        "qvae_energy_by_celltype.png",
        "qvae_energy_umap.png",
    ]
    plt.close("all")
