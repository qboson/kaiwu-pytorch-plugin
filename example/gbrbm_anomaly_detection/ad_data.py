# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""Leakage-free data preparation for unsupervised GBRBM anomaly detection.

The split is drawn on raw samples first and the ``StandardScaler`` is fitted
afterwards on the normal training rows only, so no validation or test value can
influence the scaling statistics.
"""

from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ad_config import AnomalyDatasetSplit, DataConfig


def _validate_split_inputs(
    features: np.ndarray,
    labels: np.ndarray,
    config: DataConfig,
) -> None:
    """Reject malformed matrices, labels and split ratios."""
    if not 0.0 < config.train_normal_ratio < 1.0:
        raise ValueError("train_normal_ratio must lie strictly between 0 and 1")
    if not 0.0 < config.val_ratio_of_eval < 1.0:
        raise ValueError("val_ratio_of_eval must lie strictly between 0 and 1")
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(
            "features must be a non-empty 2D array of shape (n_samples, n_features)"
        )
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("anomaly_labels must be a 1D array aligned with features")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("anomaly_labels must only contain 0 (normal) and 1 (anomaly)")
    if not np.isfinite(features).all():
        raise ValueError("features must contain only finite values")
    counts = np.bincount(labels.astype(np.int64), minlength=2)
    if counts.min() < 2:
        raise ValueError("both the normal and the anomaly class need at least two samples")


def _split_indices(
    labels: np.ndarray,
    config: DataConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split raw sample indices into normal-only train, validation and test sets."""
    normal_idx = np.flatnonzero(labels == 0)
    anomaly_idx = np.flatnonzero(labels == 1)
    train_idx, eval_normal_idx = train_test_split(
        normal_idx,
        train_size=config.train_normal_ratio,
        random_state=config.seed,
    )
    eval_idx = np.concatenate([eval_normal_idx, anomaly_idx])
    val_idx, test_idx = train_test_split(
        eval_idx,
        train_size=config.val_ratio_of_eval,
        random_state=config.seed,
        stratify=labels[eval_idx],
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def split_and_scale_dataset(
    features: np.ndarray,
    anomaly_labels: np.ndarray,
    feature_names: tuple[str, ...],
    config: DataConfig,
) -> AnomalyDatasetSplit:
    """Split raw samples first, then fit the scaler on the normal training rows."""
    matrix = np.asarray(features, dtype=np.float64)
    labels = np.asarray(anomaly_labels, dtype=np.int64)
    _validate_split_inputs(matrix, labels, config)

    train_idx, val_idx, test_idx = _split_indices(labels, config)
    scaler = StandardScaler()
    scaler.fit(matrix[train_idx])

    return AnomalyDatasetSplit(
        x_train=scaler.transform(matrix[train_idx]).astype(np.float32),
        x_val=scaler.transform(matrix[val_idx]).astype(np.float32),
        y_val=labels[val_idx],
        x_test=scaler.transform(matrix[test_idx]).astype(np.float32),
        y_test=labels[test_idx],
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        feature_names=tuple(feature_names),
        scaler=scaler,
    )


def _unpack_bunch(bunch: Any) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Extract the feature matrix, raw target vector and feature names."""
    features = np.asarray(bunch.data, dtype=np.float64)
    target = np.asarray(bunch.target)
    names = tuple(str(name) for name in bunch.feature_names)
    return features, target, names


def prepare_anomaly_dataset(config: DataConfig) -> AnomalyDatasetSplit:
    """Load Breast Cancer Wisconsin and build the leakage-free splits.

    The 357 benign rows form the normal class (label ``0``) and the 212
    malignant rows form the anomaly class (label ``1``).
    """
    features, target, names = _unpack_bunch(load_breast_cancer())
    labels = np.where(target == 1, 0, 1).astype(np.int64)
    return split_and_scale_dataset(features, labels, names, config)
