# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the RBM collaborative filtering example."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from example.collaborative_filtering.baselines import (
    PopularityRecommender,
    TruncatedSVDRecommender,
)
from example.collaborative_filtering.dataset import (
    generate_synthetic_ratings,
    load_movielens_100k,
    split_user_interactions,
)
from example.collaborative_filtering.metrics import (
    compute_mae,
    compute_rmse,
    evaluate_recommender,
    ranking_metrics_at_k,
)
from example.collaborative_filtering.model import (
    LocalIsingAnnealingSampler,
    RBMCFConfig,
    RBMCollaborativeFiltering,
    build_sampler,
)

EXAMPLE_PARENT = Path(__file__).resolve().parents[1] / "example"


def test_per_user_split_has_zero_leakage_and_preserves_observations():
    """Every observed interaction maps to exactly one split, with >=1 train item per user."""
    data = generate_synthetic_ratings(
        num_users=60,
        num_items=24,
        num_factors=4,
        density=0.30,
        val_ratio=0.15,
        test_ratio=0.20,
        seed=7,
    )

    assert not np.any(data.train_mask & data.val_mask)
    assert not np.any(data.train_mask & data.test_mask)
    assert not np.any(data.val_mask & data.test_mask)

    # Every user retains at least one training observation.
    assert np.all(data.train_mask.sum(axis=1) >= 1)
    assert data.val_mask.sum() > 0
    assert data.test_mask.sum() > 0

    # Unobserved entries must be strictly zero in every split.
    assert np.all(data.train_ratings[~data.train_mask] == 0.0)
    assert np.all(data.val_ratings[~data.val_mask] == 0.0)
    assert np.all(data.test_ratings[~data.test_mask] == 0.0)

    # Determinism check for identical seed.
    data_repeat = generate_synthetic_ratings(
        num_users=60,
        num_items=24,
        num_factors=4,
        density=0.30,
        val_ratio=0.15,
        test_ratio=0.20,
        seed=7,
    )
    np.testing.assert_array_equal(data.train_ratings, data_repeat.train_ratings)
    np.testing.assert_array_equal(data.test_ratings, data_repeat.test_ratings)


def test_split_handles_sparse_users_without_dropping_train():
    """Users with 1 or 2 observations still keep at least one training item."""
    ratings = np.array(
        [
            [5.0, 0.0, 0.0, 0.0],
            [4.0, 2.0, 0.0, 0.0],
            [5.0, 4.0, 3.0, 1.0],
        ],
        dtype=np.float32,
    )
    mask = ratings > 0.0
    tr_r, tr_m, va_r, va_m, te_r, te_m = split_user_interactions(
        ratings, mask, val_ratio=0.2, test_ratio=0.2, seed=11
    )
    assert np.all(tr_m.sum(axis=1) >= 1)
    assert tr_m[0].sum() == 1 and va_m[0].sum() == 0 and te_m[0].sum() == 0
    assert tr_m[1].sum() == 1 and te_m[1].sum() == 1 and va_m[1].sum() == 0
    np.testing.assert_array_equal(tr_m | va_m | te_m, mask)
    np.testing.assert_array_equal(tr_r + va_r + te_r, ratings)


def test_movielens_loader_reads_local_u_data(tmp_path):
    """MovieLens-100K loader parses tab-separated records and respects filters."""
    u_data = tmp_path / "u.data"
    lines = []
    # 5 users x 6 items with valid 1..5 ratings, plus one sparse user to filter out.
    for u_id in range(1, 6):
        for i_id in range(1, 7):
            rating = ((u_id + i_id) % 5) + 1
            lines.append(f"{u_id}\t{i_id}\t{rating}\t881250949")
    lines.append("99\t1\t5\t881250949")
    u_data.write_text("\n".join(lines) + "\n", encoding="utf-8")

    data = load_movielens_100k(
        u_data,
        max_items=5,
        min_user_interactions=5,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=3,
    )
    assert data.num_users == 5
    assert data.num_items == 5
    assert not np.any(data.train_mask & data.val_mask)
    assert not np.any(data.train_mask & data.test_mask)
    assert not np.any(data.val_mask & data.test_mask)


def test_metrics_exact_mathematical_values_and_leakage_exclusion():
    """Verify RMSE, MAE, Precision@K, Recall@K, and NDCG@K against hand calculation."""
    y_true = np.array([1.0, 3.0, 5.0])
    y_pred = np.array([2.0, 3.0, 3.0])
    assert compute_rmse(y_true, y_pred) == pytest.approx(math.sqrt(5.0 / 3.0), rel=1e-12)
    assert compute_mae(y_true, y_pred) == pytest.approx(1.0, rel=1e-12)

    # User 0: item 0 is seen (excluded even though it has highest score 99.0).
    # Remaining candidates ranked by score: item 2 (score 5.0, relevant),
    # item 3 (score 4.0, not relevant), item 1 (score 3.0, relevant), item 4 (score 1.0).
    # Top-3 unseen = [item 2 (hit), item 3 (miss), item 1 (hit)].
    # Precision@3 = 2/3, Recall@3 = 2/2 = 1.0.
    # DCG@3 = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1.0 + 0.5 = 1.5.
    # IDCG@3 (2 relevant items) = 1/log2(2) + 1/log2(3).
    scores = np.array([[99.0, 3.0, 5.0, 4.0, 1.0]], dtype=np.float64)
    eval_ratings = np.array([[0.0, 5.0, 4.0, 2.0, 0.0]], dtype=np.float64)
    eval_mask = np.array([[False, True, True, True, False]], dtype=bool)
    exclude_mask = np.array([[True, False, False, False, False]], dtype=bool)

    rank = ranking_metrics_at_k(
        scores=scores,
        eval_ratings=eval_ratings,
        eval_mask=eval_mask,
        exclude_mask=exclude_mask,
        top_k=3,
        relevance_threshold=4.0,
    )
    expected_dcg = 1.0 / math.log2(2.0) + 1.0 / math.log2(4.0)
    expected_idcg = 1.0 / math.log2(2.0) + 1.0 / math.log2(3.0)
    assert rank["precision_at_k"] == pytest.approx(2.0 / 3.0, rel=1e-12)
    assert rank["recall_at_k"] == pytest.approx(1.0, rel=1e-12)
    assert rank["ndcg_at_k"] == pytest.approx(expected_dcg / expected_idcg, rel=1e-12)
    assert rank["evaluated_users"] == 1.0

    # Overlapping eval_mask and exclude_mask must raise ValueError.
    with pytest.raises(ValueError, match="overlap"):
        evaluate_recommender(
            predicted_ratings=scores,
            eval_ratings=eval_ratings,
            eval_mask=eval_mask,
            exclude_mask=eval_mask,
            top_k=3,
        )


def test_build_sampler_factory_supports_local_sa_kaiwu_sa_cim_and_cd(monkeypatch, tmp_path):
    """Sampler factory constructs offline Ising SA, Kaiwu SA, CIM, and Gibbs CD modes."""
    import importlib
    import kaiwu as kw
    from kaiwu.classical import SimulatedAnnealingOptimizer
    from kaiwu.torch_plugin._kaiwu_compat import PrecisionReducer

    assert isinstance(
        build_sampler(RBMCFConfig(sampler_type="local_sa")),
        LocalIsingAnnealingSampler,
    )
    assert isinstance(
        build_sampler(RBMCFConfig(sampler_type="sa")),
        SimulatedAnnealingOptimizer,
    )
    assert build_sampler(RBMCFConfig(sampler_type="cd")) is None

    # Exercise CIM construction path without credentials or cloud requests.
    cim_module = importlib.import_module(kw.cim.CIMOptimizer.__module__)
    monkeypatch.setattr(cim_module, "ensure_license", lambda: None)
    monkeypatch.setattr(
        kw.cim.CIMOptimizer, "_get_authorization", staticmethod(lambda: "test")
    )
    monkeypatch.setattr(kw.common.CheckpointManager, "save_dir", str(tmp_path))
    cim_sampler = build_sampler(
        RBMCFConfig(sampler_type="cim", cim_task_mode="sampling")
    )
    assert isinstance(cim_sampler, PrecisionReducer)

    with pytest.raises(ValueError, match="Unsupported sampler_type"):
        build_sampler(RBMCFConfig(sampler_type="invalid"))


def test_local_ising_sampler_produces_low_energy_valid_spins():
    """LocalIsingAnnealingSampler returns valid {-1, +1} spins favoring low Ising energy."""
    num_spins = 6
    # Ferromagnetic ring: aligned spins minimize E(s) = -s^T J s.
    ising = np.zeros((num_spins, num_spins), dtype=np.float64)
    for i in range(num_spins):
        nxt = (i + 1) % num_spins
        ising[i, nxt] = 2.0
        ising[nxt, i] = 2.0

    sampler = LocalIsingAnnealingSampler(
        num_reads=8, num_sweeps=25, initial_temperature=3.0, final_temperature=0.2, seed=5
    )
    spins = sampler.solve(ising)
    assert spins.shape == (8, num_spins)
    assert set(np.unique(spins).tolist()).issubset({-1.0, 1.0})
    energies = -np.einsum("ri,ij,rj->r", spins, ising, spins)
    assert float(energies.mean()) < -10.0


def test_rbm_cf_trains_converges_and_recommends_unseen_items():
    """RBM-CF with Ising annealing sampler reduces training error and beats Popularity."""
    data = generate_synthetic_ratings(
        num_users=120,
        num_items=32,
        num_factors=4,
        density=0.35,
        noise_std=0.25,
        seed=17,
    )
    config = RBMCFConfig(
        num_hidden=16,
        epochs=20,
        batch_size=32,
        learning_rate=0.6,
        momentum=0.7,
        weight_decay=2e-4,
        annealing_weight=0.02,
        sampler_type="local_sa",
        local_sa_reads=8,
        local_sa_sweeps=12,
        seed=17,
    )
    model = RBMCollaborativeFiltering(num_items=data.num_items, config=config).fit(
        data.train_ratings,
        data.train_mask,
        val_ratings=data.val_ratings,
        val_mask=data.val_mask,
    )

    assert len(model.history_) == config.epochs
    assert model.history_[-1]["train_rmse"] < model.history_[0]["train_rmse"]

    preds = model.predict_ratings(data.train_ratings, data.train_mask)
    assert preds.shape == (data.num_users, data.num_items)
    assert np.all(np.isfinite(preds))
    assert float(preds.min()) >= data.rating_min - 1e-6
    assert float(preds.max()) <= data.rating_max + 1e-6

    top_items, top_scores = model.recommend_top_k(
        data.train_ratings, data.train_mask, top_k=5
    )
    assert top_items.shape == (data.num_users, 5)
    assert top_scores.shape == (data.num_users, 5)
    # Every recommended item must be strictly unseen in training and unique per user.
    for u_idx in range(data.num_users):
        recs = top_items[u_idx]
        assert len(set(recs.tolist())) == 5
        assert not np.any(data.train_mask[u_idx, recs])
        assert np.all(np.diff(top_scores[u_idx]) <= 1e-6)

    # Compare against Popularity and TruncatedSVD on the identical holdout split.
    exclude_mask = data.train_mask | data.val_mask
    pop = PopularityRecommender().fit(data.train_ratings, data.train_mask)
    pop_metrics = evaluate_recommender(
        predicted_ratings=pop.predict_ratings(data.num_users),
        eval_ratings=data.test_ratings,
        eval_mask=data.test_mask,
        exclude_mask=exclude_mask,
        top_k=5,
        ranking_scores=pop.predict_ranking_scores(data.num_users),
    )
    svd = TruncatedSVDRecommender(n_components=4).fit(data.train_ratings, data.train_mask)
    svd_metrics = evaluate_recommender(
        predicted_ratings=svd.predict_ratings(),
        eval_ratings=data.test_ratings,
        eval_mask=data.test_mask,
        exclude_mask=exclude_mask,
        top_k=5,
    )
    rbm_metrics = evaluate_recommender(
        predicted_ratings=preds,
        eval_ratings=data.test_ratings,
        eval_mask=data.test_mask,
        exclude_mask=exclude_mask,
        top_k=5,
    )
    assert rbm_metrics["rmse"] < pop_metrics["rmse"]
    assert rbm_metrics["ndcg_at_k"] > pop_metrics["ndcg_at_k"]
    assert svd_metrics["rmse"] < pop_metrics["rmse"]


def test_cli_fast_end_to_end_smoke(tmp_path):
    """CLI --fast mode executes cleanly and emits structured JSON without dirty repo files."""
    script_path = EXAMPLE_PARENT / "collaborative_filtering" / "train_cf.py"
    output_json = tmp_path / "cf_results.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--fast",
            "--seed",
            "42",
            "--output-json",
            str(output_json),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Popularity" in proc.stdout
    assert "TruncatedSVD" in proc.stdout
    assert "RBM-CF (LOCAL_SA)" in proc.stdout
    assert output_json.is_file()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["dataset"] == "synthetic"
    assert len(payload["results"]) == 3
    for row in payload["results"]:
        assert np.isfinite(row["metrics"]["rmse"])
        assert np.isfinite(row["metrics"]["ndcg_at_k"])
