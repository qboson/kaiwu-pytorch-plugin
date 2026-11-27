# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Benchmark runner for RBM collaborative filtering and classical baselines."""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from .baselines import PopularityRecommender, TruncatedSVDRecommender
from .dataset import (
    CollaborativeFilteringData,
    generate_synthetic_ratings,
    load_movielens_100k,
)
from .metrics import evaluate_recommender
from .model import RBMCFConfig, RBMCollaborativeFiltering


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the collaborative filtering benchmark."""
    parser = argparse.ArgumentParser(
        description=(
            "Train and benchmark an RBM collaborative filtering recommender "
            "using kaiwu-pytorch-plugin against classical baselines."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "movielens"],
        default="synthetic",
        help="Dataset source (default: synthetic offline benchmark).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to local MovieLens-100K u.data file (when --dataset movielens).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap on most active items when loading MovieLens-100K.",
    )
    parser.add_argument("--num-users", type=int, default=400, help="Synthetic users.")
    parser.add_argument("--num-items", type=int, default=80, help="Synthetic items.")
    parser.add_argument("--num-factors", type=int, default=6, help="Synthetic latent factors.")
    parser.add_argument("--density", type=float, default=0.25, help="Synthetic density.")
    parser.add_argument("--num-hidden", type=int, default=24, help="RBM hidden units.")
    parser.add_argument("--epochs", type=int, default=45, help="RBM training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=0.8, help="SGD learning rate.")
    parser.add_argument(
        "--annealing-weight",
        type=float,
        default=0.02,
        help="Density-calibrated weight on Ising annealing objective in [0, 1].",
    )
    parser.add_argument("--weight-decay", type=float, default=2e-4, help="L2 weight decay.")
    parser.add_argument(
        "--sampler",
        choices=["local_sa", "sa", "cim", "cd"],
        default="local_sa",
        help=(
            "Negative-phase sampler backend: local_sa (default offline Ising SA), "
            "sa (Kaiwu SimulatedAnnealingOptimizer), cim (Kaiwu CIMOptimizer), "
            "or cd (classical Gibbs CD)."
        ),
    )
    parser.add_argument(
        "--cim-task-name",
        type=str,
        default="kpp_cf_rbm",
        help="Task name passed to CIMOptimizer when --sampler cim.",
    )
    parser.add_argument(
        "--cim-task-mode",
        type=str,
        default=None,
        help="Optional CIM task mode (quota/optimization/sample/sampling).",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K cutoff for ranking metrics.")
    parser.add_argument("--svd-rank", type=int, default=6, help="Rank for TruncatedSVD baseline.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a compact configuration for quick smoke testing.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional file path to save structured JSON benchmark results.",
    )
    return parser


def load_dataset(args: argparse.Namespace) -> CollaborativeFilteringData:
    """Load either the synthetic benchmark or a local MovieLens-100K file."""
    if args.dataset == "movielens":
        if args.data_path is None:
            raise ValueError("--data-path is required when --dataset movielens.")
        return load_movielens_100k(
            data_path=args.data_path,
            max_items=args.max_items,
            seed=args.seed,
        )

    num_users = 120 if args.fast else args.num_users
    num_items = 40 if args.fast else args.num_items
    num_factors = min(4, args.num_factors) if args.fast else args.num_factors
    return generate_synthetic_ratings(
        num_users=num_users,
        num_items=num_items,
        num_factors=num_factors,
        density=args.density,
        seed=args.seed,
    )


def format_comparison_table(results: List[Dict[str, object]], top_k: int) -> str:
    """Format model benchmark metrics into a readable fixed-width table."""
    header = (
        f"{'Model':<22} | {'RMSE':>7} | {'MAE':>7} | "
        f"{f'P@{top_k}':>7} | {f'R@{top_k}':>7} | {f'NDCG@{top_k}':>8} | {'Time(s)':>7}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for row in results:
        metrics = row["metrics"]
        lines.append(
            f"{str(row['model']):<22} | "
            f"{metrics['rmse']:7.4f} | "
            f"{metrics['mae']:7.4f} | "
            f"{metrics['precision_at_k']:7.4f} | "
            f"{metrics['recall_at_k']:7.4f} | "
            f"{metrics['ndcg_at_k']:8.4f} | "
            f"{float(row['elapsed_sec']):7.2f}"
        )
    return "\n".join(lines)


def run_benchmark(args: argparse.Namespace) -> Dict[str, object]:
    """Execute dataset preparation, baseline evaluation, and RBM-CF training."""
    data = load_dataset(args)
    exclude_mask = data.train_mask | data.val_mask
    top_k = args.top_k
    rows: List[Dict[str, object]] = []

    t0 = time.perf_counter()
    pop_model = PopularityRecommender(
        rating_min=data.rating_min,
        rating_max=data.rating_max,
    ).fit(data.train_ratings, data.train_mask)
    pop_preds = pop_model.predict_ratings(data.num_users)
    pop_rank = pop_model.predict_ranking_scores(data.num_users)
    pop_metrics = evaluate_recommender(
        predicted_ratings=pop_preds,
        eval_ratings=data.test_ratings,
        eval_mask=data.test_mask,
        exclude_mask=exclude_mask,
        top_k=top_k,
        relevance_threshold=data.relevance_threshold,
        ranking_scores=pop_rank,
    )
    rows.append(
        {"model": "Popularity", "metrics": pop_metrics, "elapsed_sec": time.perf_counter() - t0}
    )

    svd_rank = max(1, min(args.svd_rank, data.num_users - 1, data.num_items - 1))
    t0 = time.perf_counter()
    svd_model = TruncatedSVDRecommender(
        n_components=svd_rank,
        rating_min=data.rating_min,
        rating_max=data.rating_max,
    ).fit(data.train_ratings, data.train_mask)
    svd_metrics = evaluate_recommender(
        predicted_ratings=svd_model.predict_ratings(),
        eval_ratings=data.test_ratings,
        eval_mask=data.test_mask,
        exclude_mask=exclude_mask,
        top_k=top_k,
        relevance_threshold=data.relevance_threshold,
    )
    rows.append(
        {
            "model": f"TruncatedSVD (r={svd_rank})",
            "metrics": svd_metrics,
            "elapsed_sec": time.perf_counter() - t0,
        }
    )

    num_hidden = min(12, args.num_hidden) if args.fast else args.num_hidden
    epochs = min(10, args.epochs) if args.fast else args.epochs
    rbm_config = RBMCFConfig(
        num_hidden=num_hidden,
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        annealing_weight=0.0 if args.sampler == "cd" else args.annealing_weight,
        sampler_type=args.sampler,
        local_sa_reads=8 if args.fast else 16,
        local_sa_sweeps=10 if args.fast else 20,
        sa_alpha=0.95 if args.fast else 0.99,
        sa_iterations_per_t=5 if args.fast else 10,
        sa_size_limit=16 if args.fast else 32,
        cim_task_name=args.cim_task_name,
        cim_task_mode=args.cim_task_mode,
        rating_min=data.rating_min,
        rating_max=data.rating_max,
        seed=args.seed,
    )

    t0 = time.perf_counter()
    rbm_model = RBMCollaborativeFiltering(
        num_items=data.num_items,
        config=rbm_config,
    ).fit(
        data.train_ratings,
        data.train_mask,
        val_ratings=data.val_ratings,
        val_mask=data.val_mask,
    )
    rbm_preds = rbm_model.predict_ratings(data.train_ratings, data.train_mask)
    rbm_metrics = evaluate_recommender(
        predicted_ratings=rbm_preds,
        eval_ratings=data.test_ratings,
        eval_mask=data.test_mask,
        exclude_mask=exclude_mask,
        top_k=top_k,
        relevance_threshold=data.relevance_threshold,
    )
    rows.append(
        {
            "model": f"RBM-CF ({args.sampler.upper()})",
            "metrics": rbm_metrics,
            "elapsed_sec": time.perf_counter() - t0,
        }
    )

    payload: Dict[str, object] = {
        "dataset": args.dataset,
        "summary": data.summary(),
        "top_k": top_k,
        "sampler": args.sampler,
        "rbm_config": {
            "num_hidden": num_hidden,
            "epochs": epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "annealing_weight": rbm_config.annealing_weight,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "results": rows,
        "rbm_history": rbm_model.history_,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def main(argv: Optional[List[str]] = None) -> None:
    """Parse arguments, run the benchmark, and print a summary table."""
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_benchmark(args)

    summary = payload["summary"]
    print(
        f"Dataset: {payload['dataset']} | "
        f"users={summary['num_users']}, items={summary['num_items']}, "
        f"train={summary['train_interactions']}, "
        f"val={summary['val_interactions']}, "
        f"test={summary['test_interactions']} "
        f"(density={summary['density']:.2%})"
    )
    print()
    print(format_comparison_table(payload["results"], top_k=int(payload["top_k"])))
    history = payload["rbm_history"]
    if history:
        first_ep = history[0]
        last_ep = history[-1]
        print()
        print(
            f"RBM-CF train RMSE: epoch 1 = {first_ep['train_rmse']:.4f} -> "
            f"epoch {int(last_ep['epoch'])} = {last_ep['train_rmse']:.4f}"
        )
