# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""Validation-threshold selection, held-out metrics and the IsolationForest baseline.

Every model is scored with the same convention (a higher score is more
anomalous) and the decision threshold is always selected on the validation split
before it is applied once to the test split.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ad_config import AnomalyDatasetSplit, EvaluationMetrics, ScoreBundle


def _validate_scores(scores: np.ndarray, labels: np.ndarray) -> None:
    """Reject misaligned, non-finite or single-class score vectors."""
    if scores.ndim != 1 or labels.ndim != 1:
        raise ValueError("scores and labels must be one-dimensional")
    if scores.shape[0] != labels.shape[0] or scores.shape[0] == 0:
        raise ValueError("scores and labels must have the same non-zero length")
    if not np.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")
    unique = np.unique(labels)
    if not np.isin(unique, (0, 1)).all():
        raise ValueError("labels must only contain 0 (normal) and 1 (anomaly)")
    if unique.size < 2:
        raise ValueError("labels must contain both the normal and the anomaly class")


def select_best_f1_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Select the threshold that maximizes the F1 score on the given split.

    Args:
        scores (np.ndarray): Anomaly scores, higher meaning more anomalous.
        labels (np.ndarray): Binary labels, ``1`` for anomalies.

    Returns:
        tuple: ``(best_threshold, best_f1)``.

    Raises:
        ValueError: If the vectors are misaligned, non-finite or single-class.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    _validate_scores(scores, labels)

    best_threshold = float(np.min(scores))
    best_f1 = -1.0
    for candidate in np.unique(scores):
        predicted = (scores >= candidate).astype(np.int64)
        score = float(f1_score(labels, predicted, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(candidate)
    return best_threshold, best_f1


def evaluate_anomaly_scores(
    bundle: ScoreBundle,
    model_name: str,
    score_method: str,
) -> EvaluationMetrics:
    """Select the threshold on validation scores and report test metrics.

    Args:
        bundle (ScoreBundle): Validation and test scores with their labels.
        model_name (str): Name recorded in the metrics report.
        score_method (str): Scoring method recorded in the metrics report.

    Returns:
        EvaluationMetrics: ROC-AUC, PR-AUC, F1, precision, recall and the
        validation-selected threshold.

    Raises:
        ValueError: If the bundles are misaligned or single-class.
    """
    threshold, val_f1 = select_best_f1_threshold(bundle.val_scores, bundle.val_labels)
    test_scores = np.asarray(bundle.test_scores, dtype=np.float64)
    test_labels = np.asarray(bundle.test_labels)
    _validate_scores(test_scores, test_labels)

    predicted = (test_scores >= threshold).astype(np.int64)
    return EvaluationMetrics(
        model_name=model_name,
        score_method=score_method,
        roc_auc=float(roc_auc_score(test_labels, test_scores)),
        pr_auc=float(average_precision_score(test_labels, test_scores)),
        f1=float(f1_score(test_labels, predicted, zero_division=0)),
        precision=float(precision_score(test_labels, predicted, zero_division=0)),
        recall=float(recall_score(test_labels, predicted, zero_division=0)),
        val_f1=val_f1,
        threshold=threshold,
    )


def evaluate_isolation_forest_baseline(
    split: AnomalyDatasetSplit,
    seed: int,
) -> tuple[EvaluationMetrics, np.ndarray, np.ndarray]:
    """Fit IsolationForest on the same normal split and evaluate it identically.

    Args:
        split (AnomalyDatasetSplit): The shared splits produced by ``ad_data``.
        seed (int): Random state of the forest.

    Returns:
        tuple: Metrics, validation scores and test scores, all with the
        "higher is more anomalous" convention.
    """
    forest = IsolationForest(random_state=seed)
    forest.fit(split.x_train)
    val_scores = -forest.decision_function(split.x_val)
    test_scores = -forest.decision_function(split.x_test)
    bundle = ScoreBundle(
        val_scores=val_scores,
        val_labels=split.y_val,
        test_scores=test_scores,
        test_labels=split.y_test,
    )
    metrics = evaluate_anomaly_scores(bundle, "isolation_forest", "decision_function")
    return metrics, val_scores, test_scores


def _json_default(value: Any) -> Any:
    """Serialize NumPy scalars, arrays and paths for the JSON report."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_metrics_report(output_dir: Path, payload: dict[str, Any]) -> Path:
    """Write the formatted UTF-8 ``metrics.json`` report.

    Args:
        output_dir (Path): Target directory, created when missing.
        payload (dict): JSON-serializable report content.

    Returns:
        Path: The written ``metrics.json`` path.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "metrics.json"
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def metrics_to_dict(metrics: EvaluationMetrics) -> dict[str, Any]:
    """Convert metrics into a JSON-ready dictionary.

    Args:
        metrics (EvaluationMetrics): Metrics of one model and scoring method.

    Returns:
        dict: Field name to JSON-serializable value mapping.
    """
    return asdict(metrics)
