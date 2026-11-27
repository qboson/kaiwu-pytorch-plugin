# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Cardinality-constrained Markowitz QUBO and Ising Hamiltonian formulation."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from kaiwu.torch_plugin._kaiwu_compat import qubo_matrix_to_ising_matrix


@dataclass
class PortfolioQUBO:
    """QUBO and equivalent Ising representation of Markowitz portfolio selection."""

    qubo_matrix: np.ndarray
    constant_offset: float
    ising_matrix: np.ndarray
    ising_bias: float
    target_k: int
    risk_aversion: float
    penalty: float
    expected_returns: np.ndarray
    covariance: np.ndarray

    @property
    def num_assets(self) -> int:
        """Return number of binary asset decision variables."""
        return int(self.qubo_matrix.shape[0])


def build_markowitz_qubo(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    target_k: int,
    risk_aversion: float = 2.5,
    penalty: Optional[float] = None,
) -> PortfolioQUBO:
    """Build Markowitz mean-variance + exact cardinality QUBO and Ising matrices.

    Minimizes over binary selection ``x in {0, 1}^N``::

        f(x) = (gamma / K^2) x^T Sigma x - (1 / K) mu^T x + lambda * (sum(x) - K)^2

    Expanding with ``x_i^2 = x_i`` yields ``x^T Q x + lambda * K^2`` where:
    - ``Q_ii = (gamma / K^2) * Sigma_ii - (1 / K) * mu_i + lambda * (1 - 2K)``
    - ``Q_ij = (gamma / K^2) * Sigma_ij + lambda`` for ``i != j``

    The QUBO matrix is then mapped to an ``(N + 1) x (N + 1)`` Ising matrix via
    ``kaiwu.torch_plugin._kaiwu_compat.qubo_matrix_to_ising_matrix``.
    """
    mu = np.asarray(expected_returns, dtype=np.float64).ravel()
    cov = np.asarray(covariance, dtype=np.float64)
    num_assets = len(mu)
    if cov.shape != (num_assets, num_assets):
        raise ValueError("covariance must have shape (num_assets, num_assets).")
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(cov))):
        raise ValueError("expected_returns and covariance must be finite.")
    if target_k < 1 or target_k >= num_assets:
        raise ValueError(f"target_k must be in [1, {num_assets - 1}], got {target_k}.")
    if risk_aversion < 0.0:
        raise ValueError("risk_aversion must be non-negative.")

    sym_cov = 0.5 * (cov + cov.T)
    inv_k = 1.0 / float(target_k)
    risk_scale = float(risk_aversion) * (inv_k ** 2)

    if penalty is None:
        # Calibrate cardinality penalty safely above single-asset swings while
        # preserving Ising coupling dynamic range for SA and 8-bit CIM.
        max_ret_term = float(np.max(np.abs(mu))) * inv_k
        max_risk_term = float(np.max(np.sum(np.abs(sym_cov), axis=1))) * risk_scale
        penalty_val = max(0.25, 2.0 * (max_ret_term + max_risk_term))
    else:
        penalty_val = float(penalty)
        if penalty_val <= 0.0:
            raise ValueError("penalty must be strictly positive.")

    qubo = risk_scale * sym_cov + penalty_val * np.ones((num_assets, num_assets), dtype=np.float64)
    diag_adjust = -inv_k * mu - 2.0 * penalty_val * float(target_k)
    qubo[np.diag_indices(num_assets)] += diag_adjust
    qubo = 0.5 * (qubo + qubo.T)

    constant_offset = penalty_val * float(target_k ** 2)
    ising_mat, ising_bias = qubo_matrix_to_ising_matrix(qubo)
    ising_mat = np.asarray(ising_mat, dtype=np.float64)
    ising_bias = float(ising_bias)

    return PortfolioQUBO(
        qubo_matrix=qubo,
        constant_offset=constant_offset,
        ising_matrix=ising_mat,
        ising_bias=ising_bias,
        target_k=int(target_k),
        risk_aversion=float(risk_aversion),
        penalty=penalty_val,
        expected_returns=mu,
        covariance=sym_cov,
    )


def evaluate_qubo_objective(qubo: PortfolioQUBO, selection: np.ndarray) -> np.ndarray:
    """Evaluate exact QUBO objective ``x^T Q x + constant_offset`` for binary rows."""
    bits = np.atleast_2d(np.asarray(selection, dtype=np.float64))
    if bits.shape[1] != qubo.num_assets:
        raise ValueError(
            f"Expected selection width {qubo.num_assets}, got {bits.shape[1]}."
        )
    quad = np.einsum("bi,ij,bj->b", bits, qubo.qubo_matrix, bits)
    return quad + qubo.constant_offset


def evaluate_ising_energy(ising_matrix: np.ndarray, spins: np.ndarray) -> np.ndarray:
    """Evaluate Kaiwu Ising energy ``E(s) = -s^T J s`` for spin rows ``s in {-1, +1}``."""
    spin_arr = np.atleast_2d(np.asarray(spins, dtype=np.float64))
    mat = np.asarray(ising_matrix, dtype=np.float64)
    if spin_arr.shape[1] != mat.shape[0]:
        raise ValueError(
            f"Expected spin width {mat.shape[0]}, got {spin_arr.shape[1]}."
        )
    return -np.einsum("bi,ij,bj->b", spin_arr, mat, spin_arr)


def decode_ising_spins(spins: np.ndarray) -> np.ndarray:
    """Decode auxiliary-spin Ising rows ``s in {-1, +1}^{N+1}`` to binary ``x in {0, 1}^N``."""
    spin_arr = np.atleast_2d(np.asarray(spins, dtype=np.float64))
    if spin_arr.shape[1] < 2:
        raise ValueError("Ising spin rows must include at least one variable and auxiliary spin.")
    return ((spin_arr[:, :-1] * spin_arr[:, -1:] + 1.0) * 0.5 > 0.5).astype(np.int64)


def verify_qubo_ising_equivalence(
    qubo: PortfolioQUBO,
    num_samples: int = 128,
    seed: int = 0,
) -> float:
    """Verify ``x^T Q x == -s^T J s + ising_bias`` for both auxiliary spin polarities.

    Returns:
        Maximum absolute discrepancy across all tested binary states and both
        auxiliary spin signs ``s_aux in {-1, +1}``.
    """
    rng = np.random.default_rng(seed)
    num_assets = qubo.num_assets
    if num_assets <= 10:
        grid = np.array(
            [list(map(int, format(idx, f"0{num_assets}b"))) for idx in range(1 << num_assets)],
            dtype=np.float64,
        )
    else:
        grid = rng.integers(0, 2, size=(num_samples, num_assets)).astype(np.float64)

    qubo_quad = np.einsum("bi,ij,bj->b", grid, qubo.qubo_matrix, grid)
    max_err = 0.0
    for aux_sign in (1.0, -1.0):
        var_spins = aux_sign * (2.0 * grid - 1.0)
        aux_col = np.full((grid.shape[0], 1), aux_sign, dtype=np.float64)
        full_spins = np.hstack([var_spins, aux_col])
        ising_vals = evaluate_ising_energy(qubo.ising_matrix, full_spins) + qubo.ising_bias
        max_err = max(max_err, float(np.max(np.abs(qubo_quad - ising_vals))))
    return max_err
