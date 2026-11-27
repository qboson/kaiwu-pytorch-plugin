# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""Non-blocking figure export for the anomaly detection case.

The module uses the object-oriented ``Figure`` + ``FigureCanvasAgg`` API only:
no ``pyplot`` global state, no GUI window and never a blocking ``plt.show()``.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

NORMAL_COLOR = "#2f6f9f"
ANOMALY_COLOR = "#c0392b"
THRESHOLD_COLOR = "#1a1a1a"
RANDOM_COLOR = "#8a8a8a"
CURVE_COLORS = ("#c0392b", "#2f6f9f", "#2e8b57")
NUM_BINS = 40


@dataclass(frozen=True)
class RocCurve:
    """A single ROC curve with its display name and AUC."""

    name: str
    false_positive_rate: np.ndarray
    true_positive_rate: np.ndarray
    roc_auc: float


@dataclass(frozen=True)
class PlotPayload:
    """Everything required to render the two output figures."""

    normal_scores: np.ndarray
    anomaly_scores: np.ndarray
    threshold: float
    roc_curves: tuple[RocCurve, ...]
    output_dir: Path


def _score_range(normal_scores: np.ndarray, anomaly_scores: np.ndarray) -> tuple[float, float]:
    """Return a non-degenerate histogram range covering both score groups."""
    low = float(min(np.min(normal_scores), np.min(anomaly_scores)))
    high = float(max(np.max(normal_scores), np.max(anomaly_scores)))
    if high <= low:
        high = low + 1.0
    return low, high


def save_score_distribution(payload: PlotPayload) -> Path:
    """Save the score histogram annotated with the validation threshold.

    Args:
        payload (PlotPayload): Test scores, threshold and output directory.

    Returns:
        Path: The written ``score_distribution.png`` path.
    """
    figure = Figure(figsize=(7.0, 4.5), dpi=120)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    low, high = _score_range(payload.normal_scores, payload.anomaly_scores)
    bins = np.linspace(low, high, NUM_BINS)
    axis.hist(
        payload.normal_scores,
        bins=bins,
        alpha=0.70,
        color=NORMAL_COLOR,
        label="Normal (test)",
    )
    axis.hist(
        payload.anomaly_scores,
        bins=bins,
        alpha=0.70,
        color=ANOMALY_COLOR,
        label="Anomaly (test)",
    )
    axis.axvline(
        payload.threshold,
        color=THRESHOLD_COLOR,
        linestyle="--",
        linewidth=1.2,
        label=f"Threshold = {payload.threshold:.4f}",
    )
    axis.set_xlabel("Anomaly score (higher means more anomalous)")
    axis.set_ylabel("Sample count")
    axis.set_title("GBRBM anomaly score distribution on the held-out test split")
    axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    path = Path(payload.output_dir) / "score_distribution.png"
    figure.savefig(path, format="png")
    return path


def save_roc_comparison(payload: PlotPayload) -> Path:
    """Save the ROC comparison chart against the random baseline.

    Args:
        payload (PlotPayload): ROC curves, threshold and output directory.

    Returns:
        Path: The written ``roc_curve_comparison.png`` path.
    """
    figure = Figure(figsize=(5.8, 5.0), dpi=120)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    for curve, color in zip(payload.roc_curves, CURVE_COLORS):
        axis.plot(
            curve.false_positive_rate,
            curve.true_positive_rate,
            color=color,
            linewidth=1.6,
            label=f"{curve.name} (AUC = {curve.roc_auc:.4f})",
        )
    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=RANDOM_COLOR,
        linestyle=":",
        linewidth=1.0,
        label="Random (AUC = 0.5000)",
    )
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("ROC comparison on the held-out test split")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.legend(loc="lower right", fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = Path(payload.output_dir) / "roc_curve_comparison.png"
    figure.savefig(path, format="png")
    return path


def save_anomaly_plots(payload: PlotPayload) -> tuple[Path, Path]:
    """Save both figures of the case.

    Args:
        payload (PlotPayload): Test scores, curves, threshold and output directory.

    Returns:
        tuple: Paths of ``score_distribution.png`` and
        ``roc_curve_comparison.png``.
    """
    Path(payload.output_dir).mkdir(parents=True, exist_ok=True)
    return save_score_distribution(payload), save_roc_comparison(payload)
