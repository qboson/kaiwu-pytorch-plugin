# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Classical portfolio baselines and exact brute-force QUBO optimal solver."""

import itertools
import math

import numpy as np

from .optimizer import PortfolioSolution
from .qubo import PortfolioQUBO, evaluate_qubo_objective


def equal_weight_portfolio(num_assets: int) -> np.ndarray:
    """Return the naive 1/N equal-weight portfolio vector."""
    if num_assets < 1:
        raise ValueError("num_assets must be at least 1.")
    return np.full(num_assets, 1.0 / float(num_assets), dtype=np.float64)


def greedy_topk_sharpe_portfolio(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    target_k: int,
) -> PortfolioSolution:
    """Select the top-K assets by individual in-sample Sharpe ratio (ignoring correlations)."""
    mu = np.asarray(expected_returns, dtype=np.float64).ravel()
    cov = np.asarray(covariance, dtype=np.float64)
    num_assets = len(mu)
    if target_k < 1 or target_k > num_assets:
        raise ValueError(f"target_k must lie in [1, {num_assets}].")

    vols = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    indiv_sharpe = mu / vols
    chosen = np.argsort(-indiv_sharpe, kind="mergesort")[:target_k]
    selection = np.zeros(num_assets, dtype=np.int64)
    selection[chosen] = 1
    weights = selection.astype(np.float64) / float(target_k)
    return PortfolioSolution(
        selection=selection,
        weights=weights,
        qubo_objective=float("nan"),
        feasible=True,
        num_candidates=num_assets,
    )


def exact_bruteforce_qubo_portfolio(
    qubo: PortfolioQUBO,
    max_combinations: int = 100_000,
) -> PortfolioSolution:
    """Enumerate all C(N, K) feasible subsets to obtain the exact global QUBO optimum."""
    num_assets = qubo.num_assets
    target_k = qubo.target_k
    total_combos = math.comb(num_assets, target_k)
    if total_combos > max_combinations:
        raise ValueError(
            f"C({num_assets}, {target_k}) = {total_combos} exceeds max_combinations="
            f"{max_combinations}; increase limit or reduce problem size."
        )

    best_obj = float("inf")
    best_sel = np.zeros(num_assets, dtype=np.int64)

    batch_size = 2048
    combo_iter = itertools.combinations(range(num_assets), target_k)
    while True:
        chunk = list(itertools.islice(combo_iter, batch_size))
        if not chunk:
            break
        bits = np.zeros((len(chunk), num_assets), dtype=np.float64)
        for row_idx, indices in enumerate(chunk):
            bits[row_idx, list(indices)] = 1.0
        objs = evaluate_qubo_objective(qubo, bits)
        local_best = int(np.argmin(objs))
        if float(objs[local_best]) < best_obj:
            best_obj = float(objs[local_best])
            best_sel = bits[local_best].astype(np.int64)

    weights = best_sel.astype(np.float64) / float(target_k)
    return PortfolioSolution(
        selection=best_sel,
        weights=weights,
        qubo_objective=best_obj,
        feasible=True,
        num_candidates=total_combos,
    )
