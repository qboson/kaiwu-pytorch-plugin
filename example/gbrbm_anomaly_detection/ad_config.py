# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""Frozen configuration objects and result containers for the GBRBM case.

Keeping every hyper-parameter in a ``@dataclass(frozen=True)`` holds the helper
functions below the repository pylint argument limits and guarantees that no
mutable configuration state is shared between pipeline stages.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DataConfig:
    """Data splitting configuration for normal-only unsupervised training."""

    train_normal_ratio: float = 0.6
    val_ratio_of_eval: float = 0.5
    seed: int = 42


@dataclass(frozen=True)
class ModelConfig:
    """GBRBM architecture and numerical stability settings."""

    num_hidden: int = 16
    freeze_log_var: bool = True
    eps: float = 1e-8


@dataclass(frozen=True)
class SamplerConfig:
    """Local SA default and Kaiwu 1.3.1-1.4.3 compatible CIM settings."""

    sampler_type: str = "sa"
    sa_initial_temperature: float = 100.0
    sa_alpha: float = 0.95
    sa_cutoff_temperature: float = 1.0
    sa_iterations_per_t: int = 10
    sa_size_limit: int = 32
    seed: int = 42
    cim_task_name: str = "gbrbm_anomaly_detection"
    cim_task_mode: str | None = "sampling"
    cim_wait: bool = True
    cim_sample_number: int = 32
    cim_project_no: str | None = None
    cim_use_precision_reducer: bool = True
    cim_precision: int = 8
    cim_truncated_precision: int = 10
    cim_target_bits: int = 550
    cim_tmp_dir: str | None = None


@dataclass(frozen=True)
class TrainConfig:
    """Unsupervised GBRBM training hyper-parameters."""

    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 5e-3
    momentum: float = 0.5
    weight_decay: float = 1e-2
    grad_clip_norm: float = 2.0
    seed: int = 42
    device: str = "auto"


@dataclass(frozen=True)
class AnomalyDatasetSplit:
    """Standardized splits plus the lineage indices and the fitted scaler."""

    x_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    feature_names: tuple[str, ...]
    scaler: StandardScaler


@dataclass(frozen=True)
class ScoreBundle:
    """Validation and test scores paired with their binary labels."""

    val_scores: np.ndarray
    val_labels: np.ndarray
    test_scores: np.ndarray
    test_labels: np.ndarray


@dataclass(frozen=True)
class EvaluationMetrics:
    """Held-out test metrics using a validation-selected threshold."""

    model_name: str
    score_method: str
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    val_f1: float
    threshold: float
