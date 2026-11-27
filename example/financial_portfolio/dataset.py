# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Market return generation, optional CSV loading, and chronological splitting."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PortfolioData:
    """Chronologically split asset returns (strictly zero look-ahead bias)."""

    asset_names: List[str]
    train_returns: np.ndarray
    test_returns: np.ndarray
    trading_days_per_year: int = 252
    sector_labels: Optional[np.ndarray] = None

    @property
    def num_assets(self) -> int:
        """Return number of assets in the investment universe."""
        return int(self.train_returns.shape[1])

    def estimate_train_moments(self) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate annualized mean returns and covariance strictly on training data."""
        ann_mu = (
            np.mean(self.train_returns, axis=0, dtype=np.float64)
            * float(self.trading_days_per_year)
        )
        ann_cov = (
            np.cov(self.train_returns, rowvar=False, ddof=1)
            * float(self.trading_days_per_year)
        )
        ann_cov = 0.5 * (ann_cov + ann_cov.T)
        return ann_mu, ann_cov

    def summary(self) -> dict:
        """Return basic dataset dimensions and split metadata."""
        return {
            "num_assets": self.num_assets,
            "train_days": int(self.train_returns.shape[0]),
            "test_days": int(self.test_returns.shape[0]),
            "total_days": int(self.train_returns.shape[0] + self.test_returns.shape[0]),
            "trading_days_per_year": self.trading_days_per_year,
        }


def split_returns_chronologically(
    returns: np.ndarray,
    train_ratio: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a time-ordered return matrix into past training and future test windows.

    No shuffling is performed so future returns can never leak into in-sample
    return/covariance estimation or RBM scenario training.
    """
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 20 or arr.shape[1] < 2:
        raise ValueError("returns must be a 2D array with at least 20 days and 2 assets.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns must contain only finite numeric values.")
    if not 0.2 <= train_ratio <= 0.9:
        raise ValueError("train_ratio must lie in [0.2, 0.9].")

    split_idx = int(round(arr.shape[0] * train_ratio))
    split_idx = max(10, min(arr.shape[0] - 10, split_idx))
    return arr[:split_idx].copy(), arr[split_idx:].copy()


def _simulate_regime_series(rng: np.random.Generator, num_days: int) -> np.ndarray:
    """Simulate a 2-state Markov bull/bear volatility regime path."""
    regimes = np.zeros(num_days, dtype=np.int64)
    state = 0
    for day in range(num_days):
        regimes[day] = state
        # Persistent bull (0) and volatile stress/bear (1) regimes.
        stay_prob = 0.97 if state == 0 else 0.92
        if rng.random() > stay_prob:
            state = 1 - state
    return regimes


def generate_synthetic_market_returns(
    num_assets: int = 16,
    num_sectors: int = 4,
    num_days: int = 504,
    train_ratio: float = 0.6,
    trading_days_per_year: int = 252,
    seed: int = 42,
) -> PortfolioData:
    """Generate deterministic multi-sector correlated asset returns with regimes.

    Assets belong to latent sectors driven by a shared market factor, sector
    factors, bull/stress volatility regimes, and idiosyncratic noise. Some
    sectors contain tightly correlated high-drift assets so naive greedy top-K
    selection over-concentrates risk whereas Markowitz QUBO/Ising optimization
    diversifies across sectors.
    """
    if num_assets < 4:
        raise ValueError("num_assets must be at least 4.")
    if num_sectors < 2 or num_sectors > num_assets:
        raise ValueError("num_sectors must lie in [2, num_assets].")
    if num_days < 40:
        raise ValueError("num_days must be at least 40.")

    rng = np.random.default_rng(seed)
    sector_labels = np.arange(num_assets, dtype=np.int64) % num_sectors
    sector_names = ["TECH", "HLTH", "ENRG", "FINL", "CONS", "INDU", "UTIL", "MATR"]
    asset_names = [
        f"{sector_names[sec % len(sector_names)]}_{idx:02d}"
        for idx, sec in enumerate(sector_labels)
    ]

    regimes = _simulate_regime_series(rng, num_days)
    vol_mult = np.where(regimes == 0, 1.0, 1.85)[:, None]
    regime_drift = np.where(regimes == 0, 0.00035, -0.00045)[:, None]

    # Shared market factor plus sector-specific factors (distinct risk/return profiles).
    market_factor = rng.normal(0.0, 0.0065, size=(num_days, 1)) * vol_mult
    sector_scales = np.linspace(0.0155, 0.0075, num_sectors)
    sector_factors = rng.normal(0.0, 1.0, size=(num_days, num_sectors))
    sector_factors = sector_factors * sector_scales[None, :] * vol_mult

    # Sector 0 has high drift AND high intra-sector correlation (greedy trap);
    # remaining sectors offer diverse risk-return trade-offs.
    sector_ann_drifts = np.linspace(0.15, 0.075, num_sectors)
    daily_returns = np.zeros((num_days, num_assets), dtype=np.float64)

    for idx in range(num_assets):
        sec = int(sector_labels[idx])
        beta_mkt = 0.75 + 0.35 * (sec == 0) + float(rng.uniform(-0.15, 0.15))
        beta_sec = 1.15 if sec == 0 else float(rng.uniform(0.65, 0.95))
        idio_vol = float(rng.uniform(0.0055, 0.0095))
        asset_ann_alpha = sector_ann_drifts[sec] + float(rng.uniform(-0.025, 0.035))
        daily_alpha = asset_ann_alpha / float(trading_days_per_year)

        idio_shock = rng.normal(0.0, idio_vol, size=num_days) * vol_mult[:, 0]
        daily_returns[:, idx] = (
            daily_alpha
            + regime_drift[:, 0] * beta_mkt
            + beta_mkt * market_factor[:, 0]
            + beta_sec * sector_factors[:, sec]
            + idio_shock
        )

    daily_returns = np.clip(daily_returns, -0.35, 0.35)
    train_ret, test_ret = split_returns_chronologically(daily_returns, train_ratio=train_ratio)

    return PortfolioData(
        asset_names=asset_names,
        train_returns=train_ret,
        test_returns=test_ret,
        trading_days_per_year=trading_days_per_year,
        sector_labels=sector_labels,
    )


def _parse_numeric_csv(path: Path) -> Tuple[List[str], np.ndarray]:
    """Read a headered CSV of asset prices or returns using only stdlib csv."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found at '{path}'.")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = list(csv.reader(handle))
    rows = [row for row in reader if row and any(cell.strip() for cell in row)]
    if len(rows) < 22:
        raise ValueError(f"CSV '{path}' must contain a header and at least 21 data rows.")

    header = [cell.strip() for cell in rows[0]]
    data_rows = rows[1:]
    # Automatically skip a leading non-numeric date/index column if present.
    start_col = 0
    try:
        float(data_rows[0][0])
    except ValueError:
        start_col = 1

    asset_names = header[start_col:]
    if len(asset_names) < 2:
        raise ValueError("CSV must contain at least 2 numeric asset columns.")

    values = np.zeros((len(data_rows), len(asset_names)), dtype=np.float64)
    for row_idx, row in enumerate(data_rows):
        if len(row) < start_col + len(asset_names):
            raise ValueError(f"Row {row_idx + 2} in '{path}' has fewer columns than header.")
        for col_idx, col_name in enumerate(asset_names):
            val = float(row[start_col + col_idx])
            if not np.isfinite(val):
                raise ValueError(
                    f"Non-finite value at row {row_idx + 2}, column '{col_name}'."
                )
            values[row_idx, col_idx] = val
    return asset_names, values


def load_returns_csv(
    csv_path: str | Path,
    train_ratio: float = 0.6,
    is_prices: bool = False,
    trading_days_per_year: int = 252,
) -> PortfolioData:
    """Load local CSV asset returns (or prices) and split chronologically."""
    asset_names, matrix = _parse_numeric_csv(Path(csv_path))
    if is_prices:
        if np.any(matrix <= 0.0):
            raise ValueError("Price series must be strictly positive.")
        returns = np.diff(matrix, axis=0) / matrix[:-1]
    else:
        returns = matrix

    train_ret, test_ret = split_returns_chronologically(returns, train_ratio=train_ratio)
    return PortfolioData(
        asset_names=asset_names,
        train_returns=train_ret,
        test_returns=test_ret,
        trading_days_per_year=trading_days_per_year,
    )
