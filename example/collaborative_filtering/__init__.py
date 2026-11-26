# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""RBM-based collaborative filtering example package."""

from .baselines import PopularityRecommender, TruncatedSVDRecommender
from .dataset import (
    CollaborativeFilteringData,
    generate_synthetic_ratings,
    load_movielens_100k,
    split_user_interactions,
)
from .metrics import (
    compute_mae,
    compute_rmse,
    evaluate_recommender,
    ranking_metrics_at_k,
)
from .model import (
    LocalIsingAnnealingSampler,
    RBMCFConfig,
    RBMCollaborativeFiltering,
    build_sampler,
)

__all__ = [
    "CollaborativeFilteringData",
    "LocalIsingAnnealingSampler",
    "PopularityRecommender",
    "RBMCFConfig",
    "RBMCollaborativeFiltering",
    "TruncatedSVDRecommender",
    "build_sampler",
    "compute_mae",
    "compute_rmse",
    "evaluate_recommender",
    "generate_synthetic_ratings",
    "load_movielens_100k",
    "ranking_metrics_at_k",
    "split_user_interactions",
]
