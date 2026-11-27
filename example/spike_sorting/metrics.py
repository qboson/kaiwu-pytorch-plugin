# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Clustering metrics used by the spike-sorting reproduction.

The reproduced paper reports three external metrics (ARI, AMI, V-Measure) and
three internal metrics (Calinski-Harabasz, Davies-Bouldin, Silhouette). All of
them are computed on the k-means partition of the extracted features.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
    v_measure_score,
)


def cluster_features(
    features: np.ndarray,
    n_clusters: int,
    seed: int = 0,
) -> np.ndarray:
    """Run k-means on the extracted features.

    Args:
        features: ``(n_spikes, latent_dim)`` feature matrix.
        n_clusters: number of clusters to find (the ground-truth unit count).
        seed: random seed for k-means initialisation.

    Returns:
        Integer cluster assignment per spike.
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return kmeans.fit_predict(features)


def evaluate_clustering(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = 0,
) -> dict:
    """Cluster the features and return the six reported metrics.

    Args:
        features: ``(n_spikes, latent_dim)`` feature matrix.
        labels: ground-truth unit labels (0 = multi-unit cluster).
        seed: random seed for k-means.

    Returns:
        Mapping with ``ari``, ``ami``, ``v_measure``, ``chs``, ``dbs`` and
        ``silhouette`` values, plus the number of clusters used.
    """
    n_clusters = int(len(np.unique(labels)))
    assigned = cluster_features(features, n_clusters=n_clusters, seed=seed)
    return {
        "n_clusters": n_clusters,
        "ari": float(adjusted_rand_score(labels, assigned)),
        "ami": float(adjusted_mutual_info_score(labels, assigned)),
        "v_measure": float(v_measure_score(labels, assigned)),
        "chs": float(calinski_harabasz_score(features, assigned)),
        "dbs": float(davies_bouldin_score(features, assigned)),
        "silhouette": float(silhouette_score(features, assigned)),
    }
