# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Quantum-inspired financial portfolio optimization example package."""

from .baselines import (
    equal_weight_portfolio,
    exact_bruteforce_qubo_portfolio,
    greedy_topk_sharpe_portfolio,
)
from .dataset import (
    PortfolioData,
    generate_synthetic_market_returns,
    load_returns_csv,
    split_returns_chronologically,
)
from .metrics import (
    compute_optimality_gap,
    evaluate_portfolio_out_of_sample,
)
from .optimizer import (
    LocalIsingPortfolioSampler,
    PortfolioOptimizerConfig,
    PortfolioSolution,
    RBMScenarioEnhancer,
    build_portfolio_sampler,
    solve_portfolio_ising,
)
from .qubo import (
    PortfolioQUBO,
    build_markowitz_qubo,
    decode_ising_spins,
    evaluate_ising_energy,
    evaluate_qubo_objective,
    verify_qubo_ising_equivalence,
)

__all__ = [
    "LocalIsingPortfolioSampler",
    "PortfolioData",
    "PortfolioOptimizerConfig",
    "PortfolioQUBO",
    "PortfolioSolution",
    "RBMScenarioEnhancer",
    "build_markowitz_qubo",
    "build_portfolio_sampler",
    "compute_optimality_gap",
    "decode_ising_spins",
    "equal_weight_portfolio",
    "evaluate_ising_energy",
    "evaluate_portfolio_out_of_sample",
    "evaluate_qubo_objective",
    "exact_bruteforce_qubo_portfolio",
    "generate_synthetic_market_returns",
    "greedy_topk_sharpe_portfolio",
    "load_returns_csv",
    "solve_portfolio_ising",
    "split_returns_chronologically",
    "verify_qubo_ising_equivalence",
]
