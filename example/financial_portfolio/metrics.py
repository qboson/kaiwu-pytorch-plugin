# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Out-of-sample portfolio performance and QUBO optimality-gap metrics."""

from typing import Dict, Optional

import numpy as np


def compute_optimality_gap(
    candidate_objective: float,
    optimal_objective: float,
) -> float:
    """Compute non-negative relative optimality gap vs exact global optimum."""
    diff = max(0.0, float(candidate_objective) - float(optimal_objective))
    denom = max(abs(float(optimal_objective)), 1e-12)
    return float(diff / denom)


def evaluate_portfolio_out_of_sample(
    weights: np.ndarray,
    test_returns: np.ndarray,
    target_k: Optional[int] = None,
    trading_days_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Compute annualized return, volatility, Sharpe ratio, drawdown, and feasibility."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    rets = np.asarray(test_returns, dtype=np.float64)
    if rets.ndim != 2 or rets.shape[1] != len(w):
        raise ValueError("weights length must match test_returns column count.")
    if rets.shape[0] < 2:
        raise ValueError("test_returns must contain at least 2 periods.")

    port_daily = rets @ w
    num_days = float(len(port_daily))
    ann_factor = float(trading_days_per_year)

    wealth = np.cumprod(1.0 + port_daily)
    if float(wealth[-1]) > 0.0:
        ann_return = float(wealth[-1] ** (ann_factor / num_days) - 1.0)
    else:
        ann_return = -1.0

    daily_vol = float(np.std(port_daily, ddof=1))
    ann_vol = daily_vol * float(np.sqrt(ann_factor))
    mean_ann_return = float(np.mean(port_daily)) * ann_factor
    sharpe = (
        float((mean_ann_return - risk_free_rate) / ann_vol)
        if ann_vol > 1e-12
        else 0.0
    )

    running_peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    drawdowns = (running_peak - wealth) / np.maximum(running_peak, 1e-12)
    max_drawdown = float(np.max(np.maximum(drawdowns, 0.0)))

    num_selected = int(np.sum(w > 1e-9))
    feasible = (
        float(num_selected == target_k)
        if target_k is not None
        else 1.0
    )

    return {
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "num_selected": float(num_selected),
        "cardinality_feasible": feasible,
    }
