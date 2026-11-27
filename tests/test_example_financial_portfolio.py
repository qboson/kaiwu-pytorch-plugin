# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the quantum-inspired financial portfolio optimization example."""

import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

import kaiwu as kw
import numpy as np
import pytest
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin._kaiwu_compat import PrecisionReducer

from example.financial_portfolio.baselines import (
    equal_weight_portfolio,
    exact_bruteforce_qubo_portfolio,
    greedy_topk_sharpe_portfolio,
)
from example.financial_portfolio.dataset import (
    generate_synthetic_market_returns,
    load_returns_csv,
    split_returns_chronologically,
)
from example.financial_portfolio.metrics import (
    compute_optimality_gap,
    evaluate_portfolio_out_of_sample,
)
from example.financial_portfolio.optimizer import (
    LocalIsingPortfolioSampler,
    PortfolioOptimizerConfig,
    RBMScenarioEnhancer,
    build_portfolio_sampler,
    solve_portfolio_ising,
)
from example.financial_portfolio.qubo import (
    build_markowitz_qubo,
    decode_ising_spins,
    evaluate_qubo_objective,
    verify_qubo_ising_equivalence,
)

EXAMPLE_PARENT = Path(__file__).resolve().parents[1] / "example"


def test_chronological_split_preserves_time_order_and_blocks_lookahead():
    """Chronological split never shuffles future days into the training window."""
    # Construct strictly increasing time index values so any shuffle is caught.
    days = np.arange(100, dtype=np.float64)[:, None] * 1e-4 + np.arange(4) * 1e-6
    train_r, test_r = split_returns_chronologically(days, train_ratio=0.65)
    assert train_r.shape == (65, 4)
    assert test_r.shape == (35, 4)
    assert float(train_r[:, 0].max()) < float(test_r[:, 0].min())

    # Mutating future test returns must have zero effect on training moments.
    data = generate_synthetic_market_returns(
        num_assets=10, num_sectors=3, num_days=160, train_ratio=0.6, seed=9
    )
    mu_before, cov_before = data.estimate_train_moments()
    data.test_returns[:] = 0.25
    mu_after, cov_after = data.estimate_train_moments()
    np.testing.assert_array_equal(mu_before, mu_after)
    np.testing.assert_array_equal(cov_before, cov_after)


def test_csv_loader_supports_returns_and_prices_with_date_column(tmp_path):
    """CSV loader parses optional date column, price-to-return conversion, and splits cleanly."""
    csv_file = tmp_path / "prices.csv"
    header = "Date,AAA,BBB,CCC"
    lines = [header]
    price_a, price_b, price_c = 100.0, 50.0, 200.0
    for day in range(50):
        price_a *= 1.001 + 0.0005 * ((day % 3) - 1)
        price_b *= 1.0005 + 0.0003 * ((day % 2) - 0.5)
        price_c *= 1.002 - 0.0004 * (day % 4)
        lines.append(f"2025-01-{day + 1:02d},{price_a:.6f},{price_b:.6f},{price_c:.6f}")
    csv_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    data = load_returns_csv(csv_file, train_ratio=0.6, is_prices=True)
    assert data.asset_names == ["AAA", "BBB", "CCC"]
    assert data.num_assets == 3
    assert data.train_returns.shape[0] + data.test_returns.shape[0] == 49
    assert np.all(np.isfinite(data.train_returns))
    assert np.all(np.isfinite(data.test_returns))


def test_qubo_to_ising_equivalence_holds_exhaustively_for_both_aux_polarities():
    """Exact QUBO objective matches Ising energy + bias across all 2^N states (< 1e-10)."""
    data = generate_synthetic_market_returns(
        num_assets=8, num_sectors=3, num_days=120, train_ratio=0.6, seed=13
    )
    mu, cov = data.estimate_train_moments()
    qubo = build_markowitz_qubo(mu, cov, target_k=3, risk_aversion=2.0)

    # Exhaustive verification over all 2^8 = 256 binary states x 2 aux spin signs.
    max_err = verify_qubo_ising_equivalence(qubo)
    assert max_err < 1e-10

    # Verify explicit formula matches Markowitz + cardinality penalty term-by-term.
    selection = np.array([1, 0, 1, 0, 1, 0, 0, 0], dtype=np.float64)
    w = selection / 3.0
    expected_markowitz = 2.0 * float(w @ cov @ w) - float(mu @ w)
    actual = float(evaluate_qubo_objective(qubo, selection)[0])
    assert actual == pytest.approx(expected_markowitz, rel=1e-12, abs=1e-12)

    # Infeasible cardinality (sum=2 instead of K=3) adds exact lambda * (2 - 3)^2.
    infeasible = np.array([1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float64)
    expected_infeasible = (
        2.0 * float((infeasible / 3.0) @ cov @ (infeasible / 3.0))
        - float(mu @ (infeasible / 3.0))
        + qubo.penalty * 1.0
    )
    assert float(evaluate_qubo_objective(qubo, infeasible)[0]) == pytest.approx(
        expected_infeasible, rel=1e-12, abs=1e-12
    )

    # Spin decoding is invariant under global sign flip s -> -s.
    spins = np.array([[1, -1, 1, -1, 1, -1, -1, -1, 1]], dtype=np.float64)
    np.testing.assert_array_equal(decode_ising_spins(spins), decode_ising_spins(-spins))


def test_metrics_hand_verified_return_vol_sharpe_and_drawdown():
    """Verify out-of-sample portfolio metrics against exact analytical calculations."""
    # Single-asset daily returns: +10%, -20%, +10% -> wealth: 1.1, 0.88, 0.968.
    # Peak before trough is 1.1 -> drawdown = (1.1 - 0.88) / 1.1 = 0.2.
    rets = np.array([[0.10, 0.0], [-0.20, 0.0], [0.10, 0.0]], dtype=np.float64)
    weights = np.array([1.0, 0.0], dtype=np.float64)
    metrics = evaluate_portfolio_out_of_sample(
        weights=weights, test_returns=rets, target_k=1, trading_days_per_year=252
    )
    expected_wealth = 1.1 * 0.8 * 1.1
    expected_cagr = expected_wealth ** (252.0 / 3.0) - 1.0
    expected_vol = float(np.std([0.10, -0.20, 0.10], ddof=1) * math.sqrt(252.0))
    assert metrics["annualized_return"] == pytest.approx(expected_cagr, rel=1e-12)
    assert metrics["annualized_volatility"] == pytest.approx(expected_vol, rel=1e-12)
    assert metrics["max_drawdown"] == pytest.approx(0.2, rel=1e-12)
    assert metrics["sharpe_ratio"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["cardinality_feasible"] == 1.0
    assert compute_optimality_gap(-0.08, -0.10) == pytest.approx(0.2, rel=1e-12)


def test_sampler_factory_supports_local_sa_kaiwu_sa_and_cim(monkeypatch, tmp_path):
    """Sampler factory constructs offline Ising SA, Kaiwu SA, and CIM + PrecisionReducer."""
    assert isinstance(
        build_portfolio_sampler(PortfolioOptimizerConfig(sampler_type="local_sa")),
        LocalIsingPortfolioSampler,
    )
    assert isinstance(
        build_portfolio_sampler(PortfolioOptimizerConfig(sampler_type="sa")),
        SimulatedAnnealingOptimizer,
    )

    cim_module = importlib.import_module(kw.cim.CIMOptimizer.__module__)
    monkeypatch.setattr(cim_module, "ensure_license", lambda: None)
    monkeypatch.setattr(
        kw.cim.CIMOptimizer, "_get_authorization", staticmethod(lambda: "test")
    )
    monkeypatch.setattr(kw.common.CheckpointManager, "save_dir", str(tmp_path))
    cim_sampler = build_portfolio_sampler(
        PortfolioOptimizerConfig(sampler_type="cim", cim_task_mode="optimization")
    )
    assert isinstance(cim_sampler, PrecisionReducer)

    with pytest.raises(ValueError, match="Unsupported sampler_type"):
        build_portfolio_sampler(PortfolioOptimizerConfig(sampler_type="unknown"))


def test_ising_solver_achieves_near_zero_optimality_gap_and_rbm_enhancer_runs():
    """Ising SA finds feasible exact/near-exact Markowitz optimum and RBM enhancer produces PSD cov."""
    data = generate_synthetic_market_returns(
        num_assets=12, num_sectors=4, num_days=240, train_ratio=0.6, seed=42
    )
    mu, cov = data.estimate_train_moments()
    target_k = 4
    qubo = build_markowitz_qubo(mu, cov, target_k=target_k, risk_aversion=2.5)

    exact_sol = exact_bruteforce_qubo_portfolio(qubo)
    cfg = PortfolioOptimizerConfig(
        sampler_type="local_sa",
        local_sa_reads=32,
        local_sa_sweeps=40,
        rbm_hidden=6,
        rbm_epochs=10,
        rbm_blend=0.20,
        seed=42,
    )
    ising_sol = solve_portfolio_ising(qubo, build_portfolio_sampler(cfg))
    assert ising_sol.feasible
    assert int(ising_sol.selection.sum()) == target_k
    assert float(ising_sol.weights.sum()) == pytest.approx(1.0, rel=1e-12)
    gap = compute_optimality_gap(ising_sol.qubo_objective, exact_sol.qubo_objective)
    assert gap < 0.02

    # Check baselines are valid and distinct.
    eq_w = equal_weight_portfolio(data.num_assets)
    assert float(eq_w.sum()) == pytest.approx(1.0)
    greedy_sol = greedy_topk_sharpe_portfolio(mu, cov, target_k)
    assert int(greedy_sol.selection.sum()) == target_k

    # Verify RBM scenario covariance enhancer produces finite, symmetric, PSD matrix.
    enhancer = RBMScenarioEnhancer(data.num_assets, cfg)
    enhanced_cov, rbm_stats = enhancer.fit_and_enhance_covariance(data.train_returns, cov)
    assert enhanced_cov.shape == cov.shape
    np.testing.assert_allclose(enhanced_cov, enhanced_cov.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(enhanced_cov)
    assert float(eigvals.min()) >= -1e-10
    assert rbm_stats["rbm_generated_scenarios"] >= 1.0


def test_cli_fast_end_to_end_smoke(tmp_path):
    """CLI --fast mode executes cleanly and exports valid JSON without dirty repo files."""
    script_path = EXAMPLE_PARENT / "financial_portfolio" / "run_portfolio.py"
    output_json = tmp_path / "portfolio_results.json"

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
    assert "Equal-Weight" in proc.stdout
    assert "Greedy Top-K Sharpe" in proc.stdout
    assert "Exact Brute-Force" in proc.stdout
    assert "QUBO-Ising (LOCAL_SA)" in proc.stdout
    assert "QUBO-Ising+RBM (LOCAL_SA)" in proc.stdout
    assert output_json.is_file()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["qubo_ising_max_abs_error"] < 1e-10
    assert len(payload["results"]) == 5
    for row in payload["results"]:
        assert np.isfinite(row["metrics"]["sharpe_ratio"])
        assert np.isfinite(row["metrics"]["max_drawdown"])
