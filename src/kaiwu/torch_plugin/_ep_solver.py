# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Exhaustive and Kaiwu simulated-annealing backends for EP QUBOs."""

from __future__ import annotations

import numpy as np
import torch


class ExhaustiveBackend:  # pylint: disable=too-few-public-methods
    """Enumerate every logical-bit assignment and keep an exact minimiser.

    Args:
        max_bits (int): Largest problem this backend will enumerate. Larger
            problems raise before any assignment is built.
    """

    def __init__(self, max_bits=16):
        self.max_bits = int(max_bits)

    def minimize(self, matrix):
        """Return one bit vector that minimises ``q^T Q q``.

        Args:
            matrix (torch.Tensor): Square QUBO matrix.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The best ``q`` and its objective.
            Ties keep the lowest integer assignment.

        Raises:
            ValueError: If the matrix is wider than :attr:`max_bits`.
        """
        n_bits = int(matrix.shape[0])
        if n_bits > self.max_bits:
            raise ValueError(
                f"logical bit count {n_bits} exceeds exhaustive_max_bits {self.max_bits}"
            )
        if n_bits == 0:
            empty = torch.zeros(0, dtype=matrix.dtype, device=matrix.device)
            return empty, matrix.new_zeros(())
        masks = torch.arange(1 << n_bits, device=matrix.device)
        shifts = torch.arange(n_bits, device=matrix.device)
        bits = ((masks.unsqueeze(1) >> shifts) & 1).to(dtype=matrix.dtype)
        objectives = torch.einsum("bi,ij,bj->b", bits, matrix, bits)
        best = int(torch.argmin(objectives).item())
        return bits[best], objectives[best]


AUXILIARY_SPIN = "__spin__"


class SimulatedAnnealingBackend:  # pylint: disable=too-few-public-methods
    """Minimise one QUBO with Kaiwu 1.3.1 simulated annealing.

    The annealer sees the Ising matrix from ``kaiwu.conversion``. Ranking and
    the auxiliary-spin gauge fix happen here, because 1.3.1 has no
    ``get_sorted_solutions``. ``kaiwu`` is imported on the first solve.

    Args:
        initial_temperature (float): Starting temperature.
        alpha (float): Geometric cooling factor.
        cutoff_temperature (float): Temperature at which annealing stops.
        iterations_per_t (int): Sweeps at each temperature.
        candidate_count (int): Candidate pool size, passed as ``size_limit``.
        seed (int): Paired seed forwarded to the optimizer and the solve call.
    """

    def __init__(
        self,
        initial_temperature=1.25,
        alpha=0.95,
        cutoff_temperature=0.001,
        iterations_per_t=10,
        candidate_count=8,
        seed=0,
    ):
        self.initial_temperature = float(initial_temperature)
        self.alpha = float(alpha)
        self.cutoff_temperature = float(cutoff_temperature)
        self.iterations_per_t = int(iterations_per_t)
        self.candidate_count = int(candidate_count)
        self.seed = int(seed)

    def minimize(self, matrix, init_solution=None):
        """Anneal ``matrix`` and return the best logical-bit vector.

        Args:
            matrix (torch.Tensor): Upper-triangular QUBO.
            init_solution (np.ndarray, optional): Warm start in solver order,
                length ``n + 1``, entries ``-1`` or ``+1``. ``None`` starts cold.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Bits ``q`` and ``q^T Q q``.

        Raises:
            RuntimeError: If the Ising model has no trailing ``__spin__``, or
                every anneal attempt returns an empty pool.
        """
        kaiwu = _import_kaiwu()
        qubo = np.ascontiguousarray(matrix.detach().cpu().numpy(), dtype=np.float64)
        ising = _ising_matrix(kaiwu, qubo)
        rows = _anneal(kaiwu, ising, self, init_solution)
        best = _rank_and_fix_gauge(ising, rows)[0]
        bits = torch.tensor((best[:-1] + 1.0) / 2.0, dtype=matrix.dtype, device=matrix.device)
        bits = torch.round(bits).clamp(0.0, 1.0)
        return bits, torch.dot(bits, matrix @ bits)


def _import_kaiwu():
    """Import Kaiwu on first use so the exhaustive backend stays importable."""
    import kaiwu as kw  # pylint: disable=import-outside-toplevel

    return kw


def _ising_matrix(kaiwu, qubo):
    """Convert a QUBO matrix and require the auxiliary spin to be last."""
    qubo_model = kaiwu.qubo.qubo_matrix_to_qubo_model(qubo)
    ising_model = kaiwu.conversion.qubo_model_to_ising_model(qubo_model)
    variables = ising_model.get_variables()
    n_qubo = int(qubo.shape[0])
    if len(variables) != n_qubo + 1 or variables.get(AUXILIARY_SPIN) != n_qubo:
        raise RuntimeError(
            f"Kaiwu Ising model must end with {AUXILIARY_SPIN!r} at index {n_qubo}; "
            f"got {variables!r}"
        )
    return np.ascontiguousarray(ising_model.get_matrix(), dtype=np.float64)


def _anneal(kaiwu, ising_matrix, schedule, init_solution):
    """Solve one Ising matrix, falling back when a warm start returns nothing."""
    width = int(ising_matrix.shape[0])
    attempts = [init_solution]
    if init_solution is not None:
        attempts.append(-np.asarray(init_solution, dtype=np.float64).reshape(-1))
        attempts.append(None)
    for attempt in attempts:
        start = None if attempt is None else np.ascontiguousarray(attempt, dtype=np.float64)
        try:
            raw = _one_solve(kaiwu, ising_matrix, schedule, start)
        except ValueError:
            if start is None:
                break
            continue
        rows = np.atleast_2d(np.asarray(raw, dtype=np.float64))
        if rows.ndim == 2 and rows.shape[-1] == width and rows.shape[0] > 0:
            return rows
    raise RuntimeError("Kaiwu annealer returned no candidates")


def _one_solve(kaiwu, ising_matrix, schedule, init_solution):
    """Run one single-process anneal."""
    optimizer = kaiwu.classical.SimulatedAnnealingOptimizer(
        initial_temperature=schedule.initial_temperature,
        alpha=schedule.alpha,
        cutoff_temperature=schedule.cutoff_temperature,
        iterations_per_t=schedule.iterations_per_t,
        size_limit=schedule.candidate_count,
        rand_seed=schedule.seed,
        process_num=1,
    )
    return optimizer.single_process_solve(
        ising_matrix=ising_matrix,
        init_solution=init_solution,
        rand_seed=schedule.seed,
    )


def _rank_and_fix_gauge(ising_matrix, rows):
    """Sort by ``H = -s^T J s`` and pin the auxiliary spin at ``+1``."""
    candidates = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    coupling = np.asarray(ising_matrix, dtype=np.float64)
    hamilton = -np.einsum("ij,ij->i", candidates @ coupling, candidates)
    ranked = candidates[np.argsort(hamilton)]
    return ranked * ranked[:, [-1]]
