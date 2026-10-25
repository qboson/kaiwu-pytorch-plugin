"""Contract test for ``example/qvae_cell/visualization.setup_plotting``."""

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
    scanpy.set_figure_params = lambda *args, **kwargs: None
    previous = sys.modules.get("scanpy")
    sys.modules["scanpy"] = scanpy
    try:
        spec = importlib.util.spec_from_file_location(
            "qvae_cell_visualization_setup_uut", VISUALIZATION
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


def test_configures_scanpy_matplotlib_and_seaborn():
    visualization = _load_visualization()
    scanpy_calls: list[dict[str, object]] = []
    theme_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    visualization.sc.set_figure_params = lambda **kwargs: scanpy_calls.append(kwargs)
    visualization.sns.set_theme = lambda *args, **kwargs: theme_calls.append(
        (args, kwargs)
    )
    saved = dict(plt.rcParams)
    try:
        visualization.setup_plotting()

        assert scanpy_calls == [{"dpi": 150, "frameon": False, "figsize": (6, 6)}]
        assert theme_calls == [((), {"style": "whitegrid", "palette": "muted"})]
        assert plt.rcParams["font.family"] == ["sans-serif"]
        assert plt.rcParams["font.sans-serif"] == [
            "Arial",
            "DejaVu Sans",
            "Liberation Sans",
        ]
        assert plt.rcParams["axes.unicode_minus"] is False
        assert plt.rcParams["pdf.fonttype"] == 42
    finally:
        plt.rcParams.update(saved)


def test_font_settings_are_stable_across_calls():
    visualization = _load_visualization()
    visualization.sc.set_figure_params = lambda **kwargs: None
    visualization.sns.set_theme = lambda *args, **kwargs: None
    saved = dict(plt.rcParams)
    keys = (
        "font.family",
        "font.sans-serif",
        "axes.unicode_minus",
        "pdf.fonttype",
    )
    try:
        visualization.setup_plotting()
        first = {key: plt.rcParams[key] for key in keys}
        visualization.setup_plotting()
        assert {key: plt.rcParams[key] for key in keys} == first
    finally:
        plt.rcParams.update(saved)
