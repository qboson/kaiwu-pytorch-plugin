"""14-bit adaptation that reserves resolution for the learned linear objective.

This explicitly changes the structure/objective tradeoff: lambda is rho times
the largest absolute surrogate coefficient. It is not the paper's fixed-lambda
protocol. Integer structural coefficients are represented exactly; only the
surrogate contribution is rounded.
"""
from __future__ import annotations

import numpy as np


def qubo_to_ising(Q):
    """Kaiwu's -s.T J s convention, including its auxiliary gauge spin."""
    Q = np.asarray(Q, dtype=np.float64)
    pair = (Q + Q.T) / 8.0
    diagonal = np.diag(pair).copy()
    np.fill_diagonal(pair, 0.0)
    matrix = np.zeros((len(Q) + 1, len(Q) + 1))
    matrix[:-1, :-1] = pair
    matrix[-1, :-1] = matrix[:-1, -1] = pair.sum(axis=0) + diagonal
    return -matrix, float(pair.sum() + 2 * diagonal.sum())


def prepare_balanced_qubo(structural_qubo, beta=None, penalty_ratio=4.0, bits=14):
    if not np.isfinite(penalty_ratio) or penalty_ratio <= 1:
        raise ValueError("penalty_ratio must be finite and greater than one")
    if bits != 14:
        raise ValueError("This hardware protocol requires 14-bit couplings")
    structural_qubo = np.asarray(structural_qubo, dtype=np.float64)
    n = len(structural_qubo)
    if structural_qubo.shape != (n, n) or n == 0 or not np.isfinite(structural_qubo).all():
        raise ValueError("invalid structural QUBO")
    beta = np.zeros(n) if beta is None else np.asarray(beta, dtype=np.float64)
    if beta.shape != (n,) or not np.isfinite(beta).all():
        raise ValueError("invalid surrogate coefficients")
    magnitude = float(abs(beta).max())
    lam = penalty_ratio * magnitude if magnitude else 1.0
    structure_ising, _ = qubo_to_ising(structural_qubo)
    lattice = np.rint(8 * structure_ising).astype(np.int64)
    if not np.allclose(lattice, 8 * structure_ising, atol=1e-10, rtol=0):
        raise ValueError("structure does not fit the exact eighth-integer Ising lattice")
    guide_ising, _ = qubo_to_ising(np.diag(beta))
    normalized_guide = 8 * guide_ising / lam
    limit = 8191
    largest = float(abs(lattice + normalized_guide).max())
    gain = int(np.floor(limit / largest)) if largest else 1
    if gain < 1:
        raise ValueError("structure exceeds the full-matrix 14-bit range")
    structural_integer = gain * lattice
    guide_float = gain * normalized_guide
    guide_integer = np.rint(guide_float).astype(np.int64)
    submitted = structural_integer + guide_integer
    if abs(submitted).max() > limit:
        raise ValueError("14-bit quantization overflow")
    Q = lam * structural_qubo + np.diag(beta)
    ising, offset = qubo_to_ising(Q)
    scale = 8 * gain / lam
    error = submitted / scale - ising
    guide_error = guide_integer - guide_float
    guide_norm = float(np.linalg.norm(guide_float))
    relative_guide_error = float(np.linalg.norm(guide_error) / guide_norm) if guide_norm else 0.0
    peak_integer = int(abs(guide_integer).max())
    # A nonzero objective must survive at a useful scale, not just flip a
    # structural rounding tie. Reject silently ineffective submissions.
    if magnitude and (peak_integer < 8 or relative_guide_error > 0.05):
        raise ValueError("insufficient 14-bit objective resolution; reduce penalty_ratio")
    nonzero = ising != 0
    metadata = {
        "precision_policy": "balanced_structure_exact_v1",
        "bit_width": bits, "scale": scale,
        "ising_max_abs": float(abs(ising).max()),
        "ising_offset": offset,
        "quantization_max_abs_error": float(abs(error).max()),
        "quantization_relative_frobenius_error": float(np.linalg.norm(error) / max(np.linalg.norm(ising), 1e-300)),
        "nonzero_coefficients": int(nonzero.sum()),
        "coefficients_rounded_to_zero": int((nonzero & (submitted == 0)).sum()),
        "structural_integer_exact": True, "structural_lattice_gain": gain,
        "penalty_ratio": float(penalty_ratio), "effective_lambda": lam,
        "beta_linf": magnitude, "nonzero_guidance": bool(magnitude),
        "guidance_integer_linf": peak_integer,
        "guidance_relative_error": relative_guide_error,
        "guidance_coefficients_nonzero": int((guide_float != 0).sum()),
        "guidance_coefficients_rounded_to_zero": int(((guide_float != 0) & (guide_integer == 0)).sum()),
        "fixed_lambda_protocol": False,
    }
    return Q, {"ising": submitted, "metadata": metadata,
               "structural_integer": structural_integer, "guidance_before_rounding": guide_float}
