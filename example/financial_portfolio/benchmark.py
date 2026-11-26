# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Benchmark runner and CLI parser for quantum-inspired portfolio optimization."""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .baselines import (
    equal_weight_portfolio,
    exact_bruteforce_qubo_portfolio,
    greedy_topk_sharpe_portfolio,
)
from .dataset import (
    PortfolioData,
    generate_synthetic_market_returns,
    load_returns_csv,
)
from .metrics import compute_optimality_gap, evaluate_portfolio_out_of_sample
from .optimizer import (
    PortfolioOptimizerConfig,
    RBMScenarioEnhancer,
    build_portfolio_sampler,
    solve_portfolio_ising,
)
from .qubo import (
    PortfolioQUBO,
    build_markowitz_qubo,
    evaluate_qubo_objective,
    verify_qubo_ising_equivalence,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the portfolio optimization benchmark."""
    parser = argparse.ArgumentParser(
        description=(
            "Cardinality-constrained Markowitz portfolio optimization via "
            "QUBO/Ising annealing and RBM scenario enhancement."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "csv"],
        default="synthetic",
        help="Dataset source (default: deterministic offline multi-sector synthetic).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to local CSV returns or prices file (when --dataset csv).",
    )
    parser.add_argument(
        "--is-prices",
        action="store_true",
        help="Treat CSV columns as asset prices and convert to daily returns.",
    )
    parser.add_argument("--num-assets", type=int, default=16, help="Synthetic asset count.")
    parser.add_argument("--num-sectors", type=int, default=4, help="Synthetic sector count.")
    parser.add_argument("--target-k", type=int, default=5, help="Cardinality constraint K.")
    parser.add_argument("--num-days", type=int, default=504, help="Synthetic trading days.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Chronological fraction of past days used for training.",
    )
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=2.5,
        help="Markowitz variance penalty gamma.",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=None,
        help="Optional cardinality penalty lambda (auto-calibrated if omitted).",
    )
    parser.add_argument(
        "--sampler",
        choices=["local_sa", "sa", "cim"],
        default="local_sa",
        help="Ising solver backend: local_sa (default offline), sa (Kaiwu SA), or cim.",
    )
    parser.add_argument("--rbm-hidden", type=int, default=8, help="RBM hidden units.")
    parser.add_argument("--rbm-epochs", type=int, default=25, help="RBM training epochs.")
    parser.add_argument(
        "--rbm-blend",
        type=float,
        default=0.20,
        help="Blend weight on RBM-generated scenario covariance in [0, 1].",
    )
    parser.add_argument(
        "--cim-task-name",
        type=str,
        default="kpp_portfolio_qubo",
        help="Task name passed to CIMOptimizer when --sampler cim.",
    )
    parser.add_argument(
        "--cim-task-mode",
        type=str,
        default=None,
        help="Optional CIM task mode (quota/optimization/sample/sampling).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a compact configuration for quick smoke tests.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save structured JSON benchmark results.",
    )
    return parser


def load_portfolio_dataset(args: argparse.Namespace) -> PortfolioData:
    """Load synthetic multi-sector market returns or a local CSV file."""
    if args.dataset == "csv":
        if args.data_path is None:
            raise ValueError("--data-path is required when --dataset csv.")
        return load_returns_csv(
            csv_path=args.data_path,
            train_ratio=args.train_ratio,
            is_prices=args.is_prices,
        )

    num_assets = min(12, args.num_assets) if args.fast else args.num_assets
    num_sectors = min(4, num_assets, args.num_sectors)
    num_days = min(240, args.num_days) if args.fast else args.num_days
    return generate_synthetic_market_returns(
        num_assets=num_assets,
        num_sectors=num_sectors,
        num_days=num_days,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


def _evaluate_strategy(
    name: str,
    weights: np.ndarray,
    data: PortfolioData,
    target_k: Optional[int],
    elapsed_sec: float,
    qubo_obj: Optional[float] = None,
    opt_gap: Optional[float] = None,
) -> Dict[str, object]:
    """Evaluate a portfolio strictly on future test returns and package metrics."""
    metrics = evaluate_portfolio_out_of_sample(
        weights=weights,
        test_returns=data.test_returns,
        target_k=target_k,
        trading_days_per_year=data.trading_days_per_year,
    )
    selected_assets = [
        data.asset_names[idx] for idx, w_val in enumerate(weights) if float(w_val) > 1e-9
    ]
    return {
        "strategy": name,
        "metrics": metrics,
        "qubo_objective": qubo_obj,
        "optimality_gap": opt_gap,
        "elapsed_sec": float(elapsed_sec),
        "selected_assets": selected_assets,
    }


def format_comparison_table(results: List[Dict[str, object]]) -> str:
    """Format out-of-sample portfolio results into a fixed-width table."""
    header = (
        f"{'Strategy':<28} | {'Ann.Ret':>8} | {'Ann.Vol':>8} | "
        f"{'Sharpe':>7} | {'MaxDD':>7} | {'Feas':>4} | {'Opt.Gap':>8} | {'Time(s)':>7}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for row in results:
        metrics = row["metrics"]
        feas_str = "YES" if metrics["cardinality_feasible"] >= 0.5 else "NO"
        gap_val = row["optimality_gap"]
        gap_str = f"{100.0 * float(gap_val):7.2f}%" if gap_val is not None else "    N/A "
        lines.append(
            f"{str(row['strategy']):<28} | "
            f"{100.0 * metrics['annualized_return']:7.2f}% | "
            f"{100.0 * metrics['annualized_volatility']:7.2f}% | "
            f"{metrics['sharpe_ratio']:7.3f} | "
            f"{100.0 * metrics['max_drawdown']:6.2f}% | "
            f"{feas_str:>4} | "
            f"{gap_str:>8} | "
            f"{float(row['elapsed_sec']):7.3f}"
        )
    return "\n".join(lines)


def _solve_exact_if_tractable(
    qubo: PortfolioQUBO,
    data: PortfolioData,
    target_k: int,
) -> tuple[Optional[float], Optional[Dict[str, object]]]:
    """Run exact brute-force enumeration when C(N, K) <= 100,000."""
    if math.comb(qubo.num_assets, target_k) > 100_000:
        return None, None
    t0 = time.perf_counter()
    exact_sol = exact_bruteforce_qubo_portfolio(qubo)
    row = _evaluate_strategy(
        name=f"Exact Brute-Force (K={target_k})",
        weights=exact_sol.weights,
        data=data,
        target_k=target_k,
        elapsed_sec=time.perf_counter() - t0,
        qubo_obj=exact_sol.qubo_objective,
        opt_gap=0.0,
    )
    return exact_sol.qubo_objective, row


def run_benchmark(args: argparse.Namespace) -> Dict[str, object]:
    """Run chronological split, QUBO-Ising verification, baselines, and RBM-QUBO."""
    data = load_portfolio_dataset(args)
    target_k = min(4 if args.fast else args.target_k, data.num_assets - 1)
    train_mu, train_cov = data.estimate_train_moments()

    qubo = build_markowitz_qubo(
        expected_returns=train_mu,
        covariance=train_cov,
        target_k=target_k,
        risk_aversion=args.risk_aversion,
        penalty=args.penalty,
    )
    equiv_max_err = verify_qubo_ising_equivalence(qubo, num_samples=128, seed=args.seed)

    opt_config = PortfolioOptimizerConfig(
        sampler_type=args.sampler,
        local_sa_reads=24 if args.fast else 48,
        local_sa_sweeps=30 if args.fast else 60,
        cim_task_name=args.cim_task_name,
        cim_task_mode=args.cim_task_mode,
        rbm_hidden=min(6, args.rbm_hidden) if args.fast else args.rbm_hidden,
        rbm_epochs=min(10, args.rbm_epochs) if args.fast else args.rbm_epochs,
        rbm_blend=args.rbm_blend,
        seed=args.seed,
    )

    rows: List[Dict[str, object]] = []

    # 1. Equal-Weight (1/N) baseline.
    t0 = time.perf_counter()
    eq_w = equal_weight_portfolio(data.num_assets)
    rows.append(
        _evaluate_strategy("Equal-Weight (1/N)", eq_w, data, None, time.perf_counter() - t0)
    )

    # 2. Exact Brute-Force reference (when tractable).
    exact_obj, exact_row = _solve_exact_if_tractable(qubo, data, target_k)

    # 3. Greedy Top-K Sharpe baseline.
    t0 = time.perf_counter()
    greedy_sol = greedy_topk_sharpe_portfolio(train_mu, train_cov, target_k)
    greedy_obj = float(evaluate_qubo_objective(qubo, greedy_sol.selection)[0])
    greedy_gap = (
        compute_optimality_gap(greedy_obj, exact_obj) if exact_obj is not None else None
    )
    rows.append(
        _evaluate_strategy(
            f"Greedy Top-K Sharpe (K={target_k})",
            greedy_sol.weights,
            data,
            target_k,
            time.perf_counter() - t0,
            qubo_obj=greedy_obj,
            opt_gap=greedy_gap,
        )
    )

    if exact_row is not None:
        rows.append(exact_row)

    # 4. QUBO-Ising annealing solver.
    t0 = time.perf_counter()
    ising_sampler = build_portfolio_sampler(opt_config)
    ising_sol = solve_portfolio_ising(qubo, ising_sampler)
    ising_gap = (
        compute_optimality_gap(ising_sol.qubo_objective, exact_obj)
        if exact_obj is not None
        else None
    )
    rows.append(
        _evaluate_strategy(
            f"QUBO-Ising ({args.sampler.upper()})",
            ising_sol.weights,
            data,
            target_k,
            time.perf_counter() - t0,
            qubo_obj=ising_sol.qubo_objective,
            opt_gap=ising_gap,
        )
    )

    # 5. RBM scenario-enhanced covariance + QUBO-Ising solver.
    t0 = time.perf_counter()
    rbm_enhancer = RBMScenarioEnhancer(
        num_assets=data.num_assets,
        config=opt_config,
        sampler=build_portfolio_sampler(opt_config),
    )
    rbm_cov, rbm_stats = rbm_enhancer.fit_and_enhance_covariance(
        data.train_returns, train_cov
    )
    rbm_qubo = build_markowitz_qubo(
        expected_returns=train_mu,
        covariance=rbm_cov,
        target_k=target_k,
        risk_aversion=args.risk_aversion,
        penalty=qubo.penalty,
    )
    rbm_sol = solve_portfolio_ising(rbm_qubo, build_portfolio_sampler(opt_config))
    rbm_on_base_obj = float(evaluate_qubo_objective(qubo, rbm_sol.selection)[0])
    rbm_gap = (
        compute_optimality_gap(rbm_on_base_obj, exact_obj)
        if exact_obj is not None
        else None
    )
    rows.append(
        _evaluate_strategy(
            f"QUBO-Ising+RBM ({args.sampler.upper()})",
            rbm_sol.weights,
            data,
            target_k,
            time.perf_counter() - t0,
            qubo_obj=rbm_on_base_obj,
            opt_gap=rbm_gap,
        )
    )

    payload: Dict[str, object] = {
        "dataset": args.dataset,
        "summary": data.summary(),
        "target_k": target_k,
        "risk_aversion": args.risk_aversion,
        "penalty": qubo.penalty,
        "sampler": args.sampler,
        "qubo_ising_max_abs_error": equiv_max_err,
        "rbm_stats": rbm_stats,
        "results": rows,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def main(argv: Optional[List[str]] = None) -> None:
    """Parse arguments, execute portfolio benchmark, and print comparison table."""
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_benchmark(args)

    summary = payload["summary"]
    print(
        f"Dataset: {payload['dataset']} | "
        f"assets={summary['num_assets']}, K={payload['target_k']}, "
        f"train_days={summary['train_days']}, test_days={summary['test_days']} "
        f"(gamma={payload['risk_aversion']}, lambda={payload['penalty']:.4f})"
    )
    print(
        f"QUBO-to-Ising equivalence max |error| = {payload['qubo_ising_max_abs_error']:.3e}"
    )
    print()
    print(format_comparison_table(payload["results"]))
    rbm_stats = payload["rbm_stats"]
    print()
    print(
        f"RBM scenario stats: final_obj={rbm_stats['rbm_final_objective']:.4f}, "
        f"scenarios={int(rbm_stats['rbm_generated_scenarios'])}, "
        f"joint_downside_rate={100.0 * rbm_stats['rbm_joint_downside_rate']:.2f}%"
    )
