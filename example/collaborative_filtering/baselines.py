# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Classical collaborative filtering baselines for fair benchmarking."""

from typing import Optional

import numpy as np


class PopularityRecommender:
    """Non-personalized popularity and regularized item-mean baseline."""

    def __init__(
        self,
        rating_min: float = 1.0,
        rating_max: float = 5.0,
        prior_weight: float = 5.0,
    ) -> None:
        self.rating_min = float(rating_min)
        self.rating_max = float(rating_max)
        self.prior_weight = float(prior_weight)
        self.global_mean_: float = 0.5 * (self.rating_min + self.rating_max)
        self.item_means_: Optional[np.ndarray] = None
        self.item_popularity_: Optional[np.ndarray] = None

    def fit(self, train_ratings: np.ndarray, train_mask: np.ndarray) -> "PopularityRecommender":
        """Fit item popularity counts and damped item mean ratings on training data."""
        ratings = np.asarray(train_ratings, dtype=np.float64)
        mask = np.asarray(train_mask, dtype=bool)
        if ratings.shape != mask.shape or ratings.ndim != 2:
            raise ValueError("train_ratings and train_mask must be matching 2D arrays.")
        if not np.any(mask):
            raise ValueError("train_mask must contain at least one interaction.")

        self.global_mean_ = float(ratings[mask].mean())
        item_counts = mask.sum(axis=0).astype(np.float64)
        item_sums = np.where(mask, ratings, 0.0).sum(axis=0)
        self.item_means_ = (
            item_sums + self.prior_weight * self.global_mean_
        ) / (item_counts + self.prior_weight)
        # Popularity score ranks by interaction frequency with mean rating tie-break.
        self.item_popularity_ = item_counts + 1e-3 * self.item_means_
        return self

    def predict_ratings(self, num_users: int) -> np.ndarray:
        """Broadcast damped item mean ratings for all users."""
        if self.item_means_ is None:
            raise RuntimeError("PopularityRecommender must be fit before prediction.")
        preds = np.tile(self.item_means_[None, :], (num_users, 1))
        return np.clip(preds, self.rating_min, self.rating_max).astype(np.float32)

    def predict_ranking_scores(self, num_users: int) -> np.ndarray:
        """Broadcast non-personalized popularity scores for all users."""
        if self.item_popularity_ is None:
            raise RuntimeError("PopularityRecommender must be fit before prediction.")
        return np.tile(self.item_popularity_[None, :], (num_users, 1)).astype(np.float32)


class TruncatedSVDRecommender:
    """Bias-centered Truncated SVD matrix factorization baseline.

    Learns global, user, and item biases strictly from observed training
    interactions, then extracts a low-rank truncated SVD of the centered
    residual matrix (with unobserved residuals set to zero) via NumPy.
    """

    def __init__(
        self,
        n_components: int = 8,
        bias_reg: float = 5.0,
        rating_min: float = 1.0,
        rating_max: float = 5.0,
    ) -> None:
        if n_components < 1:
            raise ValueError("n_components must be at least 1.")
        self.n_components = int(n_components)
        self.bias_reg = float(bias_reg)
        self.rating_min = float(rating_min)
        self.rating_max = float(rating_max)
        self.global_mean_: float = 0.5 * (self.rating_min + self.rating_max)
        self.user_bias_: Optional[np.ndarray] = None
        self.item_bias_: Optional[np.ndarray] = None
        self.reconstructed_residuals_: Optional[np.ndarray] = None

    def fit(self, train_ratings: np.ndarray, train_mask: np.ndarray) -> "TruncatedSVDRecommender":
        """Fit user/item biases and low-rank SVD on training interactions."""
        ratings = np.asarray(train_ratings, dtype=np.float64)
        mask = np.asarray(train_mask, dtype=bool)
        if ratings.shape != mask.shape or ratings.ndim != 2:
            raise ValueError("train_ratings and train_mask must be matching 2D arrays.")
        if not np.any(mask):
            raise ValueError("train_mask must contain at least one interaction.")

        self.global_mean_ = float(ratings[mask].mean())

        item_counts = mask.sum(axis=0).astype(np.float64)
        item_resid_sum = np.where(mask, ratings - self.global_mean_, 0.0).sum(axis=0)
        self.item_bias_ = item_resid_sum / (item_counts + self.bias_reg)

        user_counts = mask.sum(axis=1).astype(np.float64)
        user_resid_sum = np.where(
            mask, ratings - self.global_mean_ - self.item_bias_[None, :], 0.0
        ).sum(axis=1)
        self.user_bias_ = user_resid_sum / (user_counts + self.bias_reg)

        baseline = self.global_mean_ + self.user_bias_[:, None] + self.item_bias_[None, :]
        centered = np.where(mask, ratings - baseline, 0.0)

        u_mat, singular_vals, vt_mat = np.linalg.svd(centered, full_matrices=False)
        rank = min(self.n_components, len(singular_vals))
        self.reconstructed_residuals_ = (
            (u_mat[:, :rank] * singular_vals[:rank]) @ vt_mat[:rank, :]
        )
        return self

    def predict_ratings(self) -> np.ndarray:
        """Return clipped rating predictions for all user-item pairs."""
        if (
            self.user_bias_ is None
            or self.item_bias_ is None
            or self.reconstructed_residuals_ is None
        ):
            raise RuntimeError("TruncatedSVDRecommender must be fit before prediction.")

        baseline = self.global_mean_ + self.user_bias_[:, None] + self.item_bias_[None, :]
        preds = baseline + self.reconstructed_residuals_
        return np.clip(preds, self.rating_min, self.rating_max).astype(np.float32)
