# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Dataset generation, MovieLens-100K loading, and leakage-free splitting."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class CollaborativeFilteringData:
    """Container for train/validation/test user-item rating splits."""

    num_users: int
    num_items: int
    rating_min: float
    rating_max: float
    relevance_threshold: float
    train_ratings: np.ndarray
    train_mask: np.ndarray
    val_ratings: np.ndarray
    val_mask: np.ndarray
    test_ratings: np.ndarray
    test_mask: np.ndarray
    user_clusters: Optional[np.ndarray] = None
    item_categories: Optional[np.ndarray] = None

    def summary(self) -> dict:
        """Return basic interaction counts and density statistics."""
        total_cells = self.num_users * self.num_items
        train_count = int(self.train_mask.sum())
        val_count = int(self.val_mask.sum())
        test_count = int(self.test_mask.sum())
        total_obs = train_count + val_count + test_count
        return {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "train_interactions": train_count,
            "val_interactions": val_count,
            "test_interactions": test_count,
            "total_interactions": total_obs,
            "density": float(total_obs / total_cells) if total_cells > 0 else 0.0,
        }


def split_user_interactions(
    ratings: np.ndarray,
    mask: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform strict per-user holdout splitting with zero data leakage.

    Every observed interaction is assigned to exactly one of train, validation,
    or test. Each user with at least one interaction always retains at least
    one training interaction.
    """
    if ratings.shape != mask.shape or ratings.ndim != 2:
        raise ValueError("ratings and mask must be 2D arrays with identical shapes.")
    if val_ratio < 0.0 or test_ratio < 0.0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to < 1.")

    ratings_f = np.asarray(ratings, dtype=np.float32)
    mask_b = np.asarray(mask, dtype=bool)
    num_users, num_items = ratings_f.shape

    train_mask = np.zeros((num_users, num_items), dtype=bool)
    val_mask = np.zeros((num_users, num_items), dtype=bool)
    test_mask = np.zeros((num_users, num_items), dtype=bool)

    rng = np.random.default_rng(seed)

    for user_idx in range(num_users):
        observed_items = np.flatnonzero(mask_b[user_idx])
        n_obs = len(observed_items)
        if n_obs == 0:
            continue

        perm = rng.permutation(observed_items)
        if n_obs == 1:
            train_mask[user_idx, perm] = True
            continue

        n_test = int(round(n_obs * test_ratio)) if test_ratio > 0.0 else 0
        if test_ratio > 0.0 and n_test == 0 and n_obs >= 2:
            n_test = 1
        n_val = int(round(n_obs * val_ratio)) if val_ratio > 0.0 else 0
        if val_ratio > 0.0 and n_val == 0 and n_obs >= 3:
            n_val = 1

        while n_test + n_val >= n_obs:
            if n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            else:
                break

        test_items = perm[:n_test]
        val_items = perm[n_test : n_test + n_val]
        train_items = perm[n_test + n_val :]

        train_mask[user_idx, train_items] = True
        val_mask[user_idx, val_items] = True
        test_mask[user_idx, test_items] = True

    train_ratings = np.where(train_mask, ratings_f, 0.0).astype(np.float32)
    val_ratings = np.where(val_mask, ratings_f, 0.0).astype(np.float32)
    test_ratings = np.where(test_mask, ratings_f, 0.0).astype(np.float32)

    return (
        train_ratings,
        train_mask,
        val_ratings,
        val_mask,
        test_ratings,
        test_mask,
    )


def _build_synthetic_mask(
    rng: np.random.Generator,
    num_users: int,
    num_items: int,
    density: float,
) -> np.ndarray:
    """Create a sparse user-item observation mask with per-user and per-item coverage."""
    item_pop_weight = rng.uniform(0.65, 1.35, size=(1, num_items))
    obs_probs = np.clip(density * item_pop_weight, 0.05, 0.90)
    mask = rng.random((num_users, num_items)) < obs_probs

    min_per_user = max(6, min(num_items, int(round(num_items * density * 0.5))))
    for u_idx in range(num_users):
        obs_count = int(mask[u_idx].sum())
        if obs_count < min_per_user:
            missing = np.flatnonzero(~mask[u_idx])
            add_items = rng.choice(
                missing, size=min_per_user - obs_count, replace=False
            )
            mask[u_idx, add_items] = True

    for i_idx in range(num_items):
        obs_count = int(mask[:, i_idx].sum())
        if obs_count < 3:
            missing_users = np.flatnonzero(~mask[:, i_idx])
            add_users = rng.choice(
                missing_users, size=3 - obs_count, replace=False
            )
            mask[add_users, i_idx] = True

    return mask


def generate_synthetic_ratings(
    num_users: int = 400,
    num_items: int = 80,
    num_factors: int = 6,
    density: float = 0.25,
    noise_std: float = 0.35,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> CollaborativeFilteringData:
    """Generate a deterministic synthetic collaborative filtering benchmark.

    Users belong to preference clusters and items belong to latent categories
    with user/item rating biases, latent-factor affinity, and Gaussian noise,
    quantized to 1..5 star ratings. Exposure is decoupled from pure global
    quality so that popularity alone cannot solve personalized ranking.
    """
    if num_users < 10 or num_items < 10:
        raise ValueError("num_users and num_items must each be at least 10.")
    if num_factors < 2:
        raise ValueError("num_factors must be at least 2.")
    if not 0.05 <= density <= 1.0:
        raise ValueError("density must be in [0.05, 1.0].")

    rng = np.random.default_rng(seed)
    user_clusters = rng.integers(0, num_factors, size=num_users)
    item_categories = np.arange(num_items) % num_factors
    rng.shuffle(item_categories)

    user_factors = rng.normal(0.0, 0.25, size=(num_users, num_factors))
    item_factors = rng.normal(0.0, 0.25, size=(num_items, num_factors))
    for u_idx, cluster in enumerate(user_clusters):
        user_factors[u_idx, cluster] += 1.45
        user_factors[u_idx, (cluster + 1) % num_factors] += 0.45
    for i_idx, category in enumerate(item_categories):
        item_factors[i_idx, category] += 1.45
        item_factors[i_idx, (category + 1) % num_factors] += 0.35

    user_bias = rng.normal(0.0, 0.25, size=(num_users, 1))
    item_bias = rng.normal(0.0, 0.35, size=(1, num_items))

    signal = user_factors @ item_factors.T
    signal = (signal - signal.mean()) / (signal.std() + 1e-8)

    raw_ratings = (
        3.0
        + 1.35 * signal
        + user_bias
        + item_bias
        + rng.normal(0.0, noise_std, size=(num_users, num_items))
    )
    full_ratings = np.clip(np.rint(raw_ratings), 1.0, 5.0).astype(np.float32)
    mask = _build_synthetic_mask(rng, num_users, num_items, density)
    observed_ratings = np.where(mask, full_ratings, 0.0).astype(np.float32)

    splits = split_user_interactions(
        observed_ratings,
        mask,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    return CollaborativeFilteringData(
        num_users=num_users,
        num_items=num_items,
        rating_min=1.0,
        rating_max=5.0,
        relevance_threshold=4.0,
        train_ratings=splits[0],
        train_mask=splits[1],
        val_ratings=splits[2],
        val_mask=splits[3],
        test_ratings=splits[4],
        test_mask=splits[5],
        user_clusters=user_clusters,
        item_categories=item_categories,
    )


def _read_movielens_records(path: Path) -> List[Tuple[int, int, float]]:
    """Parse tab-separated MovieLens-100K ``u.data`` rating rows."""
    if not path.is_file():
        raise FileNotFoundError(
            f"MovieLens-100K file not found at '{path}'. "
            "Download ml-100k/u.data locally and pass --data-path <path>."
        )

    records: List[Tuple[int, int, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) < 3:
                raise ValueError(
                    f"Malformed MovieLens record on line {line_no}: expected "
                    f"at least 3 tab-separated columns, got {len(parts)}."
                )
            rating = float(parts[2])
            if not np.isfinite(rating) or rating <= 0.0:
                raise ValueError(
                    f"Invalid rating value {rating!r} on line {line_no}."
                )
            records.append((int(parts[0]), int(parts[1]), rating))

    if not records:
        raise ValueError(f"No valid rating records found in '{path}'.")
    return records


def load_movielens_100k(
    data_path: str | Path,
    max_items: Optional[int] = None,
    min_user_interactions: int = 10,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> CollaborativeFilteringData:
    """Load a local MovieLens-100K ``u.data`` file into leakage-free splits."""
    raw_records = _read_movielens_records(Path(data_path))

    if max_items is not None:
        if max_items < 2:
            raise ValueError("max_items must be at least 2 when specified.")
        item_counts: dict[int, int] = {}
        for _, item_id, _ in raw_records:
            item_counts[item_id] = item_counts.get(item_id, 0) + 1
        top_items = {
            item_id
            for item_id, _ in sorted(
                item_counts.items(), key=lambda kv: (-kv[1], kv[0])
            )[:max_items]
        }
        raw_records = [rec for rec in raw_records if rec[1] in top_items]

    user_counts: dict[int, int] = {}
    for user_id, _, _ in raw_records:
        user_counts[user_id] = user_counts.get(user_id, 0) + 1
    eligible_users = {
        uid for uid, count in user_counts.items() if count >= min_user_interactions
    }
    filtered = [rec for rec in raw_records if rec[0] in eligible_users]
    if not filtered:
        raise ValueError(
            "No records left after filtering with "
            f"min_user_interactions={min_user_interactions} and max_items={max_items}."
        )

    unique_users = sorted({rec[0] for rec in filtered})
    unique_items = sorted({rec[1] for rec in filtered})
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}

    ratings = np.zeros((len(unique_users), len(unique_items)), dtype=np.float32)
    mask = np.zeros((len(unique_users), len(unique_items)), dtype=bool)
    for user_id, item_id, rating in filtered:
        ratings[user_to_idx[user_id], item_to_idx[item_id]] = rating
        mask[user_to_idx[user_id], item_to_idx[item_id]] = True

    observed_values = ratings[mask]
    rating_min = float(np.min(observed_values))
    rating_max = float(np.max(observed_values))
    if rating_max <= rating_min:
        rating_min, rating_max = 1.0, 5.0

    splits = split_user_interactions(
        ratings,
        mask,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    return CollaborativeFilteringData(
        num_users=len(unique_users),
        num_items=len(unique_items),
        rating_min=rating_min,
        rating_max=rating_max,
        relevance_threshold=min(4.0, rating_max),
        train_ratings=splits[0],
        train_mask=splits[1],
        val_ratings=splits[2],
        val_mask=splits[3],
        test_ratings=splits[4],
        test_mask=splits[5],
    )
