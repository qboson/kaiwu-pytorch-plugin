# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Evaluation metrics for rating prediction and top-K recommendation."""

from typing import Dict, Optional

import numpy as np


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error on aligned rating vectors."""
    true_arr = np.asarray(y_true, dtype=np.float64).ravel()
    pred_arr = np.asarray(y_pred, dtype=np.float64).ravel()
    if true_arr.size == 0 or true_arr.shape != pred_arr.shape:
        raise ValueError("y_true and y_pred must be non-empty arrays of equal size.")
    return float(np.sqrt(np.mean((true_arr - pred_arr) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error on aligned rating vectors."""
    true_arr = np.asarray(y_true, dtype=np.float64).ravel()
    pred_arr = np.asarray(y_pred, dtype=np.float64).ravel()
    if true_arr.size == 0 or true_arr.shape != pred_arr.shape:
        raise ValueError("y_true and y_pred must be non-empty arrays of equal size.")
    return float(np.mean(np.abs(true_arr - pred_arr)))


def ranking_metrics_at_k(
    scores: np.ndarray,
    eval_ratings: np.ndarray,
    eval_mask: np.ndarray,
    exclude_mask: np.ndarray,
    top_k: int = 10,
    relevance_threshold: float = 4.0,
) -> Dict[str, float]:
    """Compute Precision@K, Recall@K, and NDCG@K strictly over unseen items.

    Items marked in ``exclude_mask`` (e.g., training interactions) are excluded
    from the candidate pool before ranking so seen items can never leak into
    top-K recommendations.

    Args:
        scores: Predicted preference scores of shape ``(num_users, num_items)``.
        eval_ratings: Held-out ground-truth ratings of shape ``(num_users, num_items)``.
        eval_mask: Boolean mask of held-out entries.
        exclude_mask: Boolean mask of already-seen items to exclude from ranking.
        top_k: Cutoff rank ``K`` (must be >= 1).
        relevance_threshold: Minimum held-out rating considered relevant.

    Returns:
        Dictionary containing ``precision_at_k``, ``recall_at_k``, ``ndcg_at_k``,
        and ``evaluated_users``.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    score_mat = np.asarray(scores, dtype=np.float64)
    eval_r = np.asarray(eval_ratings, dtype=np.float64)
    eval_m = np.asarray(eval_mask, dtype=bool)
    excl_m = np.asarray(exclude_mask, dtype=bool)

    if len({score_mat.shape, eval_r.shape, eval_m.shape, excl_m.shape}) != 1:
        raise ValueError("scores, eval_ratings, eval_mask, and exclude_mask must match in shape.")
    if score_mat.ndim != 2:
        raise ValueError("Ranking inputs must be 2D matrices of shape (num_users, num_items).")
    if np.any(eval_m & excl_m):
        raise ValueError("eval_mask and exclude_mask overlap; potential data leakage detected.")

    num_users, num_items = score_mat.shape
    precisions = []
    recalls = []
    ndcgs = []

    discounts = 1.0 / np.log2(np.arange(2, min(top_k, num_items) + 2, dtype=np.float64))

    for user_idx in range(num_users):
        relevant_mask = eval_m[user_idx] & (eval_r[user_idx] >= relevance_threshold)
        num_relevant = int(relevant_mask.sum())
        if num_relevant == 0:
            continue

        candidate_mask = ~excl_m[user_idx]
        num_candidates = int(candidate_mask.sum())
        if num_candidates == 0:
            continue

        effective_k = min(top_k, num_candidates)
        masked_scores = np.where(candidate_mask, score_mat[user_idx], -np.inf)
        # Stable descending sort ensures deterministic tie-breaking by item index.
        ranked_items = np.argsort(-masked_scores, kind="mergesort")[:effective_k]
        hits = relevant_mask[ranked_items].astype(np.float64)

        hit_count = float(hits.sum())
        precisions.append(hit_count / float(effective_k))
        recalls.append(hit_count / float(num_relevant))

        dcg = float(np.sum(hits * discounts[:effective_k]))
        ideal_hits = min(effective_k, num_relevant)
        idcg = float(np.sum(discounts[:ideal_hits]))
        ndcgs.append(dcg / idcg if idcg > 0.0 else 0.0)

    if not precisions:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "evaluated_users": 0.0,
        }

    return {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "ndcg_at_k": float(np.mean(ndcgs)),
        "evaluated_users": float(len(precisions)),
    }


def evaluate_recommender(
    predicted_ratings: np.ndarray,
    eval_ratings: np.ndarray,
    eval_mask: np.ndarray,
    exclude_mask: np.ndarray,
    top_k: int = 10,
    relevance_threshold: float = 4.0,
    ranking_scores: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluate rating accuracy (RMSE, MAE) and top-K ranking quality."""
    pred_r = np.asarray(predicted_ratings, dtype=np.float64)
    eval_r = np.asarray(eval_ratings, dtype=np.float64)
    eval_m = np.asarray(eval_mask, dtype=bool)
    excl_m = np.asarray(exclude_mask, dtype=bool)

    if len({pred_r.shape, eval_r.shape, eval_m.shape, excl_m.shape}) != 1:
        raise ValueError("All rating and mask arrays must share the same shape.")
    if np.any(eval_m & excl_m):
        raise ValueError("eval_mask and exclude_mask overlap; potential data leakage detected.")
    if not np.any(eval_m):
        raise ValueError("eval_mask must contain at least one observed entry.")

    rmse = compute_rmse(eval_r[eval_m], pred_r[eval_m])
    mae = compute_mae(eval_r[eval_m], pred_r[eval_m])

    rank_matrix = pred_r if ranking_scores is None else np.asarray(ranking_scores, dtype=np.float64)
    rank_metrics = ranking_metrics_at_k(
        scores=rank_matrix,
        eval_ratings=eval_r,
        eval_mask=eval_m,
        exclude_mask=excl_m,
        top_k=top_k,
        relevance_threshold=relevance_threshold,
    )

    return {
        "rmse": rmse,
        "mae": mae,
        **rank_metrics,
    }
