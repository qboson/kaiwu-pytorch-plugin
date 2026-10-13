# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=too-many-lines  # intentionally self-contained single-file plugin
"""
QDEQ: a reusable Deep Equilibrium (DEQ) fixed-point plugin for PyTorch.

DEQ models replace an explicit stack of N identical layers with a single
weight-tied function f_theta, solved to a fixed point z* = f_theta(z*, x)
via repeated iteration; gradients are computed with the implicit function
theorem instead of backpropagating through the iteration. ``QDEQ`` wraps an
arbitrary ``func(z, *args) -> z_next`` with this forward solve plus implicit
backward, decoupled from both the wrapped model's architecture and the
fixed-point solver used.

Two solvers are provided out of the box and share one call signature, so
either can serve as the forward or backward solver interchangeably:

- ``anderson_solver``: classical Anderson-accelerated fixed-point iteration.
- ``ising_sa_solver``: reformulates the local fixed-point update as a
  QUBO/Ising problem over binary-encoded coefficients, solved by simulated
  annealing (CPU) or a compatible physical Ising/CIM backend. This is the
  default forward solver.

Quickstart::

    import torch
    from kaiwu.torch_plugin.qdeq import QDEQ

    deq = QDEQ()  # Ising/SA forward solver, Anderson backward solver
    deq.train()
    z_out, info = deq(lambda z, x: 0.3 * z + x, torch.zeros_like(x), x)
    z_out.square().mean().backward()

If ``func`` (or the instance a bound ``func`` is attached to) implements
``set_quant(bool)``, QDEQ automatically disables it for the no-grad forward
solve and re-enables it for the single differentiable reattach call, so a
W8A8 (or similar) fake-quantized model iterates in full precision and only
quantizes the one call that participates in backpropagation. ``QDEQ.eval()``
never touches ``set_quant`` and returns the raw fixed-point approximation.

See ``example/qdeq/`` for a complete usage example, including an adapter
that attaches QDEQ to a production iTransformer decoder.
"""

from __future__ import annotations

import contextlib
import dataclasses
import time
from dataclasses import dataclass
from typing import Callable, ContextManager, Protocol

import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Ising/QUBO fixed-point solving
# ---------------------------------------------------------------------------


@dataclass
class IsingSolverConfig:  # pylint: disable=too-many-instance-attributes
    """Hyperparameters of the Ising/SA fixed-point solver."""

    solver_backend: str = "cpu_numpy"
    solver_variant: str = "simple_coeff"
    threshold: int = 12
    eps: float = 1e-3
    basis: int = 4
    history_size: int = 4
    coeff_bits: int = 3
    coeff_min: float = -1.0
    coeff_step: float = 0.5
    basis_mode: str = "mixed_history"
    fd_eps: float = 1e-3
    matrix_scale: float = 1.0
    sa_restarts: int = 8
    sa_steps: int = 80
    sa_initial_temp: float = 1.0
    sa_cooling: float = 0.95
    seed: int = 123
    reattach_mode: str = "straight_through"

    @classmethod
    def from_dict(cls, data: object) -> "IsingSolverConfig":
        if not isinstance(data, dict):
            return cls()
        valid = {field.name for field in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in valid})


class LocalAnnealingBackend:  # pylint: disable=too-few-public-methods
    """Small CPU simulated annealing backend for dense Ising matrices."""

    def __init__(self, config: IsingSolverConfig):
        self.config = config

    def solve(self, matrix: np.ndarray, seed: int) -> tuple[np.ndarray, float, float]:
        rng = np.random.default_rng(seed)
        n = matrix.shape[0]
        best_spin = None
        best_energy = float("inf")
        start = time.perf_counter()
        for _ in range(self.config.sa_restarts):
            spin = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=n)
            energy = ising_energy(matrix, spin)
            temp = self.config.sa_initial_temp
            for _step in range(self.config.sa_steps):
                for idx in rng.permutation(n):
                    flipped = spin.copy()
                    flipped[idx] *= -1.0
                    new_energy = ising_energy(matrix, flipped)
                    delta = new_energy - energy
                    if delta <= 0.0 or rng.random() < np.exp(-delta / max(temp, 1e-8)):
                        spin = flipped
                        energy = new_energy
                temp *= self.config.sa_cooling
            if energy < best_energy:
                best_energy = energy
                best_spin = spin.copy()
        return best_spin.astype(np.float32), time.perf_counter() - start, best_energy


def ising_energy(matrix: np.ndarray, spin: np.ndarray) -> float:
    """Compute s^T M s."""
    return float(spin @ matrix @ spin)


class TorchSimulatedAnnealing:  # pylint: disable=too-few-public-methods
    """GPU-friendly simulated annealing backend for dense Ising matrices."""

    def __init__(self, config: IsingSolverConfig):
        self.config = config

    def solve(self, matrix: torch.Tensor, seed: int) -> tuple[torch.Tensor, float, float]:
        device = matrix.device
        dtype = matrix.dtype
        n = matrix.shape[0]
        chains = int(self.config.sa_restarts)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        start = time.perf_counter()
        spin = torch.randint(
            0, 2, (chains, n), device=device, generator=generator, dtype=torch.int64
        ).to(dtype)
        spin = spin.mul_(2.0).sub_(1.0)
        energy = torch_ising_energy(matrix, spin)
        best_energy, best_idx = torch.min(energy, dim=0)
        best_spin = spin[best_idx].clone()
        temp = float(self.config.sa_initial_temp)
        chain_idx = torch.arange(chains, device=device)
        for _step in range(int(self.config.sa_steps)):
            for _ in range(n):
                flip_idx = torch.randint(0, n, (chains,), device=device, generator=generator)
                proposal = spin.clone()
                proposal[chain_idx, flip_idx] *= -1.0
                new_energy = torch_ising_energy(matrix, proposal)
                delta = new_energy - energy
                accept_prob = torch.exp(-delta / max(temp, 1e-8))
                uniform = torch.rand(chains, device=device, generator=generator)
                accept = (delta <= 0.0) | (uniform < accept_prob)
                spin = torch.where(accept.unsqueeze(1), proposal, spin)
                energy = torch.where(accept, new_energy, energy)
                current_best, current_idx = torch.min(energy, dim=0)
                if bool(current_best < best_energy):
                    best_energy = current_best
                    best_spin = spin[current_idx].clone()
            temp *= float(self.config.sa_cooling)
        return best_spin, time.perf_counter() - start, float(best_energy.detach().item())


def torch_ising_energy(matrix: torch.Tensor, spin: torch.Tensor) -> torch.Tensor:
    """Compute batched s^T M s on the current torch device."""
    return torch.einsum("bi,ij,bj->b", spin, matrix, spin)


def normalize_tensor(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-9)


class CoefficientIsingFixedPointSolver:  # pylint: disable=too-few-public-methods
    """Solve z = f(z) by Ising-encoded low-dimensional coefficient updates.

    At each outer iteration we build a small basis V, approximate the residual
    change r(z + Va) with finite differences, and choose binary-encoded
    coefficients by minimizing a QUBO/Ising objective.
    """

    def __init__(self, config: IsingSolverConfig):
        self.config = config
        self.backend = LocalAnnealingBackend(config)
        self.torch_backend = TorchSimulatedAnnealing(config)
        self._last_sa_time = 0.0
        self._last_best_energy = float("nan")
        self._last_predicted_energy = float("nan")
        self._last_coeff_norm = float("nan")

    @torch.no_grad()
    def solve(
        self, f: Callable[[torch.Tensor], torch.Tensor], z0: torch.Tensor
    ) -> dict[str, object]:
        z = z0.detach().clone()
        residual_history: list[torch.Tensor] = []
        update_history: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = []
        traces: dict[str, list] = {
            key: [] for key in ("abs", "rel", "rel_v5", "energy", "coeff_norm", "basis_dim")
        }
        sa_time = 0.0
        start = time.perf_counter()
        best_z = z.clone()
        best_rel = float("inf")

        for step in range(1, self.config.threshold + 1):
            residual = f(z) - z
            rel_res = float((residual.norm() / z.norm().clamp_min(1e-8)).detach().cpu())
            rel_res_v5 = float(
                (residual.norm() / (residual + z).norm().clamp_min(1e-9)).detach().cpu()
            )
            traces["abs"].append(float(residual.norm().detach().cpu()))
            traces["rel"].append(rel_res)
            traces["rel_v5"].append(rel_res_v5)
            tracked_rel = rel_res_v5 if self.config.solver_variant == "v5_dynamic" else rel_res
            if tracked_rel < best_rel:
                best_rel = tracked_rel
                best_z = z.clone()
            if tracked_rel < self.config.eps:
                break

            if self.config.solver_variant == "v5_dynamic":
                basis = self._make_basis_v5(z, residual, residual_history, update_history)
            elif self.config.solver_variant == "simple_coeff":
                basis = self._make_basis(residual, residual_history, update_history)
            else:
                raise ValueError(f"Unsupported solver_variant: {self.config.solver_variant}")
            traces["basis_dim"].append(len(basis))
            jv = self._finite_difference_jv(f, z, residual, basis)
            coeff = self._solve_coefficients(residual, jv, step)
            delta = torch.zeros_like(z)
            for c, v in zip(coeff, basis):
                delta = delta + float(c) * v
            next_z = z + delta
            r_next = f(next_z) - next_z
            if self.config.solver_variant == "v5_dynamic":
                update_history.append((delta.detach(), (r_next - residual).detach()))
            else:
                update_history.append((next_z - z).detach())
            residual_history.append(residual.detach())
            residual_history = residual_history[-self.config.history_size :]
            update_history = update_history[-self.config.history_size :]
            z = next_z
            traces["energy"].append(float(self._last_predicted_energy))
            traces["coeff_norm"].append(float(self._last_coeff_norm))
            sa_time += self._last_sa_time

        return self._final_result(f, best_z, traces, sa_time, start)

    def _final_result(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        result: torch.Tensor,
        traces: dict[str, list],
        sa_time: float,
        start: float,
    ) -> dict[str, object]:
        final_residual = f(result) - result
        final_abs = final_residual.norm()
        old_final_rel_debug = final_abs / result.norm().clamp_min(1e-8)
        final_rel_v5 = final_abs / (final_residual + result).norm().clamp_min(1e-9)
        is_v5 = self.config.solver_variant == "v5_dynamic"
        energy_trace = traces["energy"]
        return {
            "result": result,
            "abs_trace": traces["abs"],
            "rel_trace": traces["rel"],
            "rel_trace_v5": traces["rel_v5"],
            "residual_abs_trace": traces["abs"],
            "residual_rel_trace_v5": traces["rel_v5"],
            "energy_trace": energy_trace,
            "coeff_norm_trace": traces["coeff_norm"],
            "basis_dim_trace": traces["basis_dim"],
            "nstep": len(traces["abs"]),
            "final_abs": final_abs,
            "final_rel": final_rel_v5 if is_v5 else old_final_rel_debug,
            "final_rel_v5": final_rel_v5,
            "old_final_rel_debug": old_final_rel_debug,
            "solver_time": time.perf_counter() - start,
            "sa_time": sa_time,
            "energy_last": energy_trace[-1] if energy_trace else float("nan"),
            "energy_best": min(energy_trace) if energy_trace else float("nan"),
            "solver": "ising",
            "solver_variant": self.config.solver_variant,
        }

    def _make_basis(
        self,
        residual: torch.Tensor,
        residual_history: list[torch.Tensor],
        update_history: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    ) -> list[torch.Tensor]:
        basis: list[torch.Tensor] = []
        residual_norm = residual.norm().clamp_min(1e-8)
        basis.append(-residual / residual_norm)
        for update in reversed(update_history):
            basis.append(update / update.norm().clamp_min(1e-8))
            if len(basis) >= self.config.basis:
                return basis
        for old_residual in reversed(residual_history):
            direction = residual - old_residual
            basis.append(direction / direction.norm().clamp_min(1e-8))
            if len(basis) >= self.config.basis:
                return basis
        while len(basis) < self.config.basis:
            rand = torch.randn_like(residual)
            basis.append(rand / rand.norm().clamp_min(1e-8))
        return basis

    def _make_basis_v5(
        self,
        z: torch.Tensor,
        residual: torch.Tensor,
        residual_history: list[torch.Tensor],
        update_history: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    ) -> list[torch.Tensor]:
        basis: list[torch.Tensor] = [normalize_tensor(-residual)]
        if self.config.basis_mode in {"residual_history", "mixed_history"}:
            for old_residual in reversed(residual_history):
                if len(basis) >= self.config.basis:
                    break
                basis.append(normalize_tensor(old_residual))
        if self.config.basis_mode in {"update_history", "mixed_history"}:
            for item in reversed(update_history):
                if isinstance(item, tuple):
                    dz, dr = item
                else:
                    dz, dr = item, torch.zeros_like(item)
                if len(basis) < self.config.basis:
                    basis.append(normalize_tensor(dz))
                if len(basis) < self.config.basis:
                    basis.append(normalize_tensor(dr))
                if len(basis) >= self.config.basis:
                    break
        while len(basis) < self.config.basis:
            rnd = torch.randint_like(z, low=0, high=2).float() * 2.0 - 1.0
            basis.append(normalize_tensor(rnd))
        return basis[: self.config.basis]

    def _finite_difference_jv(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        z: torch.Tensor,
        residual: torch.Tensor,
        basis: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        jv = []
        eps = self.config.fd_eps
        for vector in basis:
            residual_shifted = f(z + eps * vector) - (z + eps * vector)
            jv.append((residual_shifted - residual) / eps)
        return jv

    def _record_coeff_stats(self, coeff: np.ndarray, quad: np.ndarray, lin: np.ndarray) -> None:
        self._last_predicted_energy = float(0.5 * coeff @ quad @ coeff + lin @ coeff)
        self._last_coeff_norm = float(np.linalg.norm(coeff))

    def _solve_coefficients(
        self, residual: torch.Tensor, jv: list[torch.Tensor], step: int
    ) -> np.ndarray:
        if self.config.solver_variant == "v5_dynamic":
            return self._solve_coefficients_v5(residual, jv, step)
        if self.config.solver_backend == "gpu_torch":
            return self._solve_coefficients_torch(residual, jv, step)
        if self.config.solver_backend != "cpu_numpy":
            raise ValueError(f"Unsupported solver_backend: {self.config.solver_backend}")
        basis_count = len(jv)
        bits = self.config.coeff_bits
        weights = np.array(
            [self.config.coeff_step * (2**idx) for idx in range(bits)], dtype=np.float64
        )
        offsets = np.full(basis_count, self.config.coeff_min, dtype=np.float64)
        jv_flat = torch.stack([v.reshape(-1) for v in jv], dim=1)
        residual_flat = residual.reshape(-1, 1)
        quad = (jv_flat.T @ jv_flat).detach().cpu().double().numpy()
        lin = (jv_flat.T @ residual_flat).squeeze(1).detach().cpu().double().numpy()

        n_binary = basis_count * bits
        q = np.zeros((n_binary, n_binary), dtype=np.float64)
        c = np.zeros(n_binary, dtype=np.float64)
        for i in range(basis_count):
            for bi in range(bits):
                idx_i = i * bits + bi
                c[idx_i] += lin[i] * weights[bi]
                for j in range(basis_count):
                    for bj in range(bits):
                        idx_j = j * bits + bj
                        q[idx_i, idx_j] += 0.5 * quad[i, j] * weights[bi] * weights[bj]
        for i in range(basis_count):
            for bi in range(bits):
                idx_i = i * bits + bi
                c[idx_i] += float(np.dot(quad[i], offsets)) * weights[bi]
        matrix = qubo_to_ising(q, c) * self.config.matrix_scale
        spin, elapsed, best_energy = self.backend.solve(matrix, seed=self.config.seed + step)
        self._last_sa_time = elapsed
        self._last_best_energy = best_energy
        binary = (spin > 0).astype(np.float64)
        coeff = offsets.copy()
        for i in range(basis_count):
            coeff[i] += float(np.dot(binary[i * bits : (i + 1) * bits], weights))
        self._record_coeff_stats(coeff, quad, lin)
        return coeff

    def _solve_coefficients_v5(
        self, residual: torch.Tensor, jv: list[torch.Tensor], step: int
    ) -> np.ndarray:
        if self.config.solver_backend != "cpu_numpy":
            raise ValueError("v5_dynamic currently supports only solver_backend=cpu_numpy")
        m = len(jv)
        bits = int(self.config.coeff_bits)
        r_flat = residual.reshape(-1)
        jv_flat = torch.stack([v.reshape(-1) for v in jv], dim=0)
        a_mat = 0.5 * (jv_flat @ jv_flat.t()).detach().cpu().double().numpy()
        b_vec = (jv_flat @ r_flat).detach().cpu().double().numpy()
        coeff_min, coeff_step, scale = self._v5_residual_scaled_coeff_spec(
            float(residual.norm().detach().cpu()), m, bits
        )
        offset, enc = self._v5_basis_coeff_matrix(m, coeff_min, coeff_step, bits, scale)
        matrix, n_bits = self._v5_build_matrix(a_mat, b_vec, offset, enc)
        spin, elapsed, best_energy = self.backend.solve(matrix, seed=self.config.seed + step)
        self._last_sa_time = elapsed
        self._last_best_energy = best_energy
        pred_energy, coeff = self._v5_candidate_binary_coeffs(
            spin, n_bits, offset, enc, a_mat, b_vec
        )
        self._last_predicted_energy = pred_energy
        self._last_coeff_norm = float(np.linalg.norm(coeff))
        return coeff

    @staticmethod
    def _v5_residual_scaled_coeff_spec(
        r_norm: float, m: int, bits: int
    ) -> tuple[float, float, float]:
        scale = r_norm / np.sqrt(max(1, m))
        return -2.0, 4.0 / ((2**bits) - 1), scale

    @staticmethod
    def _v5_basis_coeff_matrix(
        m: int, coeff_min: float, coeff_step: float, bits: int, scale: float
    ) -> tuple[np.ndarray, np.ndarray]:
        n = m * bits
        enc = np.zeros((m, n), dtype=np.float64)
        offset = np.full(m, coeff_min * scale, dtype=np.float64)
        for i in range(m):
            for j in range(bits):
                enc[i, i * bits + j] = scale * coeff_step * (2**j)
        return offset, enc

    def _v5_candidate_binary_coeffs(
        self,
        result: np.ndarray,
        n_bits: int,
        offset: np.ndarray,
        enc: np.ndarray,
        a_mat: np.ndarray,
        b_vec: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        arr = np.asarray(result)
        if arr.ndim == 1:
            arr = arr[None, :]
        candidates: list[tuple[float, np.ndarray]] = []
        for row in arr:
            if row.shape[0] < n_bits + 1:
                continue
            raw = np.sign(row[: n_bits + 1]).astype(np.float64)
            raw[raw == 0] = 1
            spin = raw[:n_bits] * raw[n_bits]
            x = (spin + 1.0) / 2.0
            coeff = offset + enc @ x
            energy = float(coeff @ a_mat @ coeff + b_vec @ coeff)
            candidates.append((energy, coeff))
        if not candidates:
            x = np.zeros(n_bits, dtype=np.float64)
            coeff = offset + enc @ x
            candidates.append((float(coeff @ a_mat @ coeff + b_vec @ coeff), coeff))
        candidates.sort(key=lambda item: item[0])
        return candidates[0]

    def _v5_build_matrix(
        self, a_mat: np.ndarray, b_vec: np.ndarray, offset: np.ndarray, enc: np.ndarray
    ) -> tuple[np.ndarray, int]:
        qmat = enc.T @ a_mat @ enc
        qlin = 2.0 * (offset @ a_mat @ enc) + (b_vec @ enc)
        n_bits = qmat.shape[0]
        spin_quad = 0.25 * qmat
        spin_lin = 0.5 * (qmat @ np.ones(n_bits, dtype=np.float64)) + 0.5 * qlin
        mat = np.zeros((n_bits + 1, n_bits + 1), dtype=np.float64)
        mat[:n_bits, :n_bits] = spin_quad
        mat[:n_bits, n_bits] = spin_lin / 2.0
        mat[n_bits, :n_bits] = spin_lin / 2.0
        mat = 0.5 * (mat + mat.T)
        np.fill_diagonal(mat, 0.0)
        max_abs = max(float(np.max(np.abs(mat))), 1e-12)
        return mat / max_abs * float(self.config.matrix_scale), n_bits

    def _solve_coefficients_torch(
        self, residual: torch.Tensor, jv: list[torch.Tensor], step: int
    ) -> list[float]:
        basis_count = len(jv)
        bits = int(self.config.coeff_bits)
        device = residual.device
        dtype = torch.float64
        weights = torch.tensor(
            [self.config.coeff_step * (2**idx) for idx in range(bits)], device=device, dtype=dtype
        )
        offsets = torch.full(
            (basis_count,), float(self.config.coeff_min), device=device, dtype=dtype
        )
        jv_flat = torch.stack([v.reshape(-1) for v in jv], dim=1).to(dtype)
        residual_flat = residual.reshape(-1, 1).to(dtype)
        quad = jv_flat.T @ jv_flat
        lin = (jv_flat.T @ residual_flat).squeeze(1)
        n_binary = basis_count * bits
        q = torch.zeros((n_binary, n_binary), device=device, dtype=dtype)
        c = torch.zeros(n_binary, device=device, dtype=dtype)
        for i in range(basis_count):
            for bi in range(bits):
                idx_i = i * bits + bi
                c[idx_i] = c[idx_i] + lin[i] * weights[bi]
                for j in range(basis_count):
                    for bj in range(bits):
                        idx_j = j * bits + bj
                        q[idx_i, idx_j] = (
                            q[idx_i, idx_j] + 0.5 * quad[i, j] * weights[bi] * weights[bj]
                        )
        for i in range(basis_count):
            for bi in range(bits):
                idx_i = i * bits + bi
                c[idx_i] = c[idx_i] + torch.dot(quad[i], offsets) * weights[bi]
        matrix = qubo_to_ising_torch(q, c) * float(self.config.matrix_scale)
        spin, elapsed, best_energy = self.torch_backend.solve(matrix, seed=self.config.seed + step)
        self._last_sa_time = elapsed
        self._last_best_energy = best_energy
        binary = (spin > 0).to(dtype)
        coeff = offsets.clone()
        for i in range(basis_count):
            coeff[i] = coeff[i] + torch.dot(binary[i * bits : (i + 1) * bits], weights)
        self._record_coeff_stats(
            coeff.detach().cpu().double().numpy(),
            quad.detach().cpu().double().numpy(),
            lin.detach().cpu().double().numpy(),
        )
        return [float(x.detach().item()) for x in coeff]


def qubo_to_ising(q: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Convert x in {0,1} QUBO to s in {-1,1} Ising matrix, dropping constants."""
    q = 0.5 * (q + q.T)
    n = q.shape[0]
    matrix = q / 4.0
    linear = c / 2.0 + q.sum(axis=1) / 4.0
    out = matrix.copy()
    for i in range(n):
        out[i, i] += linear[i]
    return out


def qubo_to_ising_torch(q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Convert x in {0,1} QUBO to s in {-1,1} Ising matrix on torch device."""
    q = 0.5 * (q + q.T)
    matrix = q / 4.0
    linear = c / 2.0 + q.sum(dim=1) / 4.0
    out = matrix.clone()
    idx = torch.arange(q.shape[0], device=q.device)
    out[idx, idx] = out[idx, idx] + linear
    return out


# ---------------------------------------------------------------------------
# Unified fixed-point solver interface
# ---------------------------------------------------------------------------
#
# Every solver below (and every solver a caller plugs into ``QDEQ``) shares
# the call signature::
#
#     solve(f, z0, threshold, eps=1e-3, stop_mode="rel", **kwargs) -> dict
#
# and returns a dict containing at least ``"result"`` (the solution tensor)
# and ``"nstep"`` (an int). This lets any two solvers be swapped for
# ``f_solver``/``b_solver`` without changing caller code.
#
# ``QDEQ`` omits the ``threshold`` kwarg entirely when its
# ``f_thres``/``b_thres`` is ``None`` (its default), so any solver plugged in
# here must supply its own usable default for ``threshold`` — it cannot rely
# on the caller always providing one, despite the required-looking signature
# below (``anderson_solver``'s default is ``50``; ``ising_sa_solver``'s
# falls back to its ``IsingSolverConfig.threshold``).


class Solver(Protocol):  # pylint: disable=too-few-public-methods
    """Call signature shared by every fixed-point solver plugged into ``QDEQ``."""

    def __call__(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        z0: torch.Tensor,
        threshold: int,
        eps: float = 1e-3,
        stop_mode: str = "rel",
        **kwargs: object,
    ) -> dict:
        ...


def anderson_solver(
    f: Callable[[torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    threshold: int = 50,
    eps: float = 1e-3,
    stop_mode: str = "rel",
    m: int = 6,
    lam: float = 1e-4,
    beta: float = 1.0,
    **kwargs: object,
) -> dict:
    """Anderson-accelerated fixed-point solver: finds z* such that z* = f(z*).

    Pluggable into ``QDEQ`` as either ``f_solver`` or ``b_solver``. ``z0``
    has a batch axis followed by arbitrary feature axes.
    """
    del kwargs  # accepted for ``Solver`` signature compatibility
    if z0.ndim < 2 or z0.numel() == 0:
        raise ValueError("z0 must have nonempty batch and feature dimensions")
    if threshold < 3 or m < 2:
        raise ValueError("Anderson requires threshold >= 3 and m >= 2")
    if stop_mode not in {"rel", "abs"}:
        raise ValueError("stop_mode must be 'rel' or 'abs'")
    bsz, flat_dim = z0.shape[0], z0[0].numel()
    alternative_mode = "rel" if stop_mode == "abs" else "abs"

    x_hist = torch.zeros(bsz, m, flat_dim, dtype=z0.dtype, device=z0.device)
    f_hist = torch.zeros(bsz, m, flat_dim, dtype=z0.dtype, device=z0.device)

    x_hist[:, 0], f_hist[:, 0] = z0.reshape(bsz, -1), f(z0).reshape(bsz, -1)
    x_hist[:, 1], f_hist[:, 1] = f_hist[:, 0], f(f_hist[:, 0].reshape_as(z0)).reshape(bsz, -1)

    h_mat = torch.zeros(bsz, m + 1, m + 1, dtype=z0.dtype, device=z0.device)
    h_mat[:, 0, 1:] = h_mat[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=z0.dtype, device=z0.device)
    y[:, 0] = 1

    trace_dict = {"abs": [], "rel": []}
    lowest_dict = {"abs": 1e8, "rel": 1e8}
    lowest_step_dict = {"abs": 0, "rel": 0}
    lowest_xest = z0.clone()

    for k in range(2, threshold):
        n = min(k, m)

        g_mat = f_hist[:, :n] - x_hist[:, :n]
        h_mat[:, 1 : n + 1, 1 : n + 1] = torch.bmm(
            g_mat, g_mat.transpose(1, 2)
        ) + lam * torch.eye(n, dtype=z0.dtype, device=z0.device)[None]
        alpha = torch.linalg.solve(h_mat[:, : n + 1, : n + 1], y[:, : n + 1])[:, 1 : n + 1, 0]

        x_hist[:, k % m] = beta * (alpha[:, None] @ f_hist[:, :n])[:, 0] + (1 - beta) * (
            alpha[:, None] @ x_hist[:, :n]
        )[:, 0]
        f_hist[:, k % m] = f(x_hist[:, k % m].reshape_as(z0)).reshape(bsz, -1)

        gx = (f_hist[:, k % m] - x_hist[:, k % m]).view_as(z0)
        abs_diff = gx.norm().item()
        rel_diff = abs_diff / (1e-5 + f_hist[:, k % m].norm().item())
        diff_dict = {"abs": abs_diff, "rel": rel_diff}
        trace_dict["abs"].append(abs_diff)
        trace_dict["rel"].append(rel_diff)

        for mode in ["rel", "abs"]:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode:
                    lowest_xest = x_hist[:, k % m].view_as(z0).clone().detach()
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = k

        if trace_dict[stop_mode][-1] < eps:
            for _ in range(threshold - 1 - k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break

    return {
        "result": lowest_xest,
        "lowest": lowest_dict[stop_mode],
        "nstep": lowest_step_dict[stop_mode],
        "prot_break": False,
        "abs_trace": trace_dict["abs"],
        "rel_trace": trace_dict["rel"],
        "eps": eps,
        "threshold": threshold,
    }


def ising_sa_solver(config: IsingSolverConfig) -> Solver:
    """Wrap ``CoefficientIsingFixedPointSolver`` behind the shared ``Solver`` signature.

    ``config.threshold``/``config.eps`` are the defaults; passing ``threshold``
    or ``eps`` at call time (as ``QDEQ`` does, via ``f_thres``/``b_thres``)
    overrides them for that call without mutating the original config.

    Warning: when used as ``b_solver`` (the backward/implicit-gradient
    solver), ``config.solver_variant="simple_coeff"`` (the dataclass default)
    is residual-magnitude-blind: ``qubo_to_ising`` folds the linear/residual
    term onto the Ising diagonal, where it cancels as a constant in the SA
    energy delta, so the annealer picks coefficients independent of the
    actual backward residual's size. Use
    ``config.solver_variant="v5_dynamic"`` (which self-scales its
    coefficient range to the current residual norm) when this solver serves
    as ``b_solver``; ``simple_coeff`` is fine as ``f_solver``, since forward
    iterates only need to track progress toward a stopping threshold, not
    solve a linear system exactly.
    """

    def _solve(
        f,
        z0,
        threshold: int | None = None,
        eps: float | None = None,
        stop_mode: str = "rel",
        **kwargs: object,
    ) -> dict:
        del stop_mode, kwargs  # accepted for ``Solver`` signature compatibility
        cfg = config
        if threshold is not None or eps is not None:
            cfg = dataclasses.replace(
                config,
                threshold=threshold if threshold is not None else config.threshold,
                eps=eps if eps is not None else config.eps,
            )
        return CoefficientIsingFixedPointSolver(cfg).solve(f, z0)

    return _solve


# ---------------------------------------------------------------------------
# Optional diagnostics
# ---------------------------------------------------------------------------
#
# Not called by ``QDEQ``. Call these manually on the ``(info["new_z"],
# info["z_star"])`` pair from a training-mode forward pass if you want
# jacobian regularization or a spectral-radius estimate.
#
# Call these *before* ``.backward()``, and fold their result into the loss
# you then call ``.backward()`` on. ``new_z``'s autograd graph is freed once
# that ``.backward()`` runs (normal PyTorch behavior) — calling these
# functions afterwards raises ``RuntimeError: Trying to backward through the
# graph a second time``.


def jac_loss_estimate(
    f0: torch.Tensor, z0: torch.Tensor, vecs: int = 2, create_graph: bool = True
) -> torch.Tensor:
    """Estimate tr(J^T J) = tr(J J^T) via the Hutchinson estimator.

    Args:
        f0: Output of the function f (whose Jacobian J is analyzed).
        z0: Input to f.
        vecs: Number of random Gaussian probe vectors.
        create_graph: Whether to build a backward graph (needed to train on this loss).

    Returns:
        A scalar tensor: the shape-normalized jacobian loss.
    """
    if vecs < 1:
        raise ValueError("vecs must be positive")
    result = 0
    for _ in range(vecs):
        v = torch.randn(*z0.shape).to(z0)
        vjp = torch.autograd.grad(f0, z0, v, retain_graph=True, create_graph=create_graph)[0]
        result = result + vjp.norm() ** 2
    return result / vecs / np.prod(z0.shape)


def power_method(
    f0: torch.Tensor, z0: torch.Tensor, n_iters: int = 200
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate the spectral radius of J = df0/dz0 via power iteration on J^T.

    A diagnostic, not a convergence guarantee: a dominant real eigenvalue
    and a spectral gap are needed for ordinary power iteration to converge.
    The graph is retained so the caller can subsequently backpropagate loss.

    Args:
        f0: Output of the function f (whose Jacobian J is analyzed).
        z0: Input to f.
        n_iters: Number of power-method iterations.

    Returns:
        (dominant eigenvector of J^T, estimated absolute eigenvalue).
    """
    if n_iters < 1:
        raise ValueError("n_iters must be positive")
    evector = torch.randn_like(z0)
    bsz = evector.shape[0]
    evalue = None
    tiny = torch.finfo(z0.dtype).tiny
    for _ in range(n_iters):
        vtj = torch.autograd.grad(f0, z0, evector, retain_graph=True, create_graph=False)[0]
        vector = evector.reshape(bsz, -1)
        product = vtj.reshape(bsz, -1)
        evalue = (product * vector).sum(1, keepdim=True) / vector.square().sum(
            1, keepdim=True
        ).clamp_min(tiny)
        evector = (product / product.norm(dim=1, keepdim=True).clamp_min(tiny)).reshape_as(z0)
    return evector, torch.abs(evalue)


# ---------------------------------------------------------------------------
# QDEQ: forward fixed-point solve + implicit-function-theorem backward
# ---------------------------------------------------------------------------


def _solver_kwargs(threshold: int | None, stop_mode: str, eps: float | None = None) -> dict:
    """Build kwargs for a Solver call, omitting ``threshold`` entirely when
    ``None`` so the solver's own default (or config-carried value) applies.
    """
    kwargs = {"stop_mode": stop_mode}
    if threshold is not None:
        kwargs["threshold"] = threshold
    if eps is not None:
        kwargs["eps"] = eps
    return kwargs


class _DEQImplicitGrad(torch.autograd.Function):  # pylint: disable=abstract-method
    """Identity in the forward direction; in backward, replaces the incoming
    gradient at ``new_z`` with the solution of the linear fixed-point system
    (I - J^T) y = grad_output, where J = d(new_z)/d(z_star). Autograd then
    keeps flowing this corrected gradient through ``new_z``'s own graph (i.e.
    through func's parameters and any other func_args) exactly as it would
    for an ordinary op — this is the implicit function theorem backward pass
    from Bai et al. 2019, Theorem 1.
    """

    @staticmethod
    def forward(  # pylint: disable=arguments-differ
        ctx, new_z, z_star, b_solver, b_thres, stop_mode, info, b_eps
    ):
        ctx.b_solver = b_solver
        ctx.b_thres = b_thres
        ctx.stop_mode = stop_mode
        ctx.info = info
        ctx.b_eps = b_eps
        ctx.save_for_backward(z_star, new_z)
        return new_z

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        z_star, new_z = ctx.saved_tensors

        def backward_fixed_point(y):
            vjp = torch.autograd.grad(new_z, z_star, y, retain_graph=True, allow_unused=True)[0]
            return grad_output if vjp is None else vjp + grad_output

        result = ctx.b_solver(
            backward_fixed_point,
            torch.zeros_like(grad_output),
            **_solver_kwargs(ctx.b_thres, ctx.stop_mode, ctx.b_eps),
        )
        for key, value in result.items():
            if key != "result":
                ctx.info[f"backward_{key}"] = value
        return result["result"], None, None, None, None, None, None


def _quant_target(func):
    """Resolve the object QDEQ should call ``set_quant(bool)`` on: ``func``
    itself, or — if ``func`` is a bound method — the instance it is bound
    to. ``hasattr(bound_method, "set_quant")`` is ``False`` even when the
    method's owning instance has that attribute (a bound method does not
    proxy attribute lookups to ``__self__``), so checking only ``func``
    would silently skip quantization orchestration for the "bound method"
    calling style this class's docstring documents as supported. Returns
    ``None`` if neither has a ``set_quant`` method.
    """
    if hasattr(func, "set_quant"):
        return func
    bound_to = getattr(func, "__self__", None)
    if bound_to is not None and hasattr(bound_to, "set_quant"):
        return bound_to
    return None


def _default_ising_solver() -> Solver:
    """QDEQ's default forward solver: the Ising/SA solver with
    ``solver_variant="v5_dynamic"`` (self-scales its coefficient range to the
    current residual norm), not the ``IsingSolverConfig`` dataclass's own
    default ``"simple_coeff"`` — ``simple_coeff`` is residual-magnitude-blind
    (see ``ising_sa_solver``'s docstring) and only safe as a forward-only
    solver. The backward default is independently Anderson.
    """
    return ising_sa_solver(IsingSolverConfig(solver_variant="v5_dynamic"))


class QDEQ(nn.Module):
    """Wraps an arbitrary f_theta(z, *args) with forward fixed-point solve.

    ``func`` may be any callable with signature ``func(z, *func_args) -> z_next``
    where ``z_next`` has the same shape as ``z`` — an ``nn.Module``, a bound
    method, or a plain closure.

    ``f_solver`` defaults to the Ising/SA solver (``ising_sa_solver`` with
    ``solver_variant="v5_dynamic"``); ``b_solver`` defaults to
    ``anderson_solver``, not to ``f_solver`` — pass ``f_solver=``/
    ``b_solver=`` explicitly to use something else.

    W8A8 (or similar) quantization toggling is orchestrated automatically:
    if ``func`` has a ``set_quant(bool)`` method (or is a bound method whose
    ``__self__`` does), QDEQ calls ``set_quant(False)`` before the no-grad
    forward solve (so iteration runs in full precision) and
    ``set_quant(True)`` before the single differentiable reattach call (so
    only that one call runs quantized). If the forward solve raises,
    ``set_quant(True)`` is still called before the exception propagates, so
    a failed training step never leaves quantization stuck disabled for
    later calls. Eval mode never toggles quantization at all (it is left
    exactly as the caller set it, so evaluation reflects true low-precision
    inference without QDEQ's involvement). If ``func`` has no ``set_quant``
    method, none of this applies and QDEQ is fully dtype-agnostic.
    ``solve_ctx``/``reattach_ctx`` remain available underneath for any other
    external state (dropout, etc.) that isn't covered by ``set_quant`` —
    both fire around the same two phases.

    ``func`` must be deterministic across the no-grad solve and the single
    differentiable reattach call other than through ``set_quant``/
    ``solve_ctx``/``reattach_ctx`` — e.g. put any ``nn.Module`` it wraps in
    ``eval()`` mode (or otherwise disable dropout) before calling this
    module. If ``func`` behaves differently between the two phases for any
    other reason, the implicit-function-theorem assumption
    (``new_z ≈ z_star``) breaks and both the forward residual and the
    backward gradients silently degrade.

    Example — a decoder implementing ``set_quant`` gets full-precision
    iterate / quantized reattach automatically, with the default Ising
    forward solver. ``QDEQ()`` defaults to training mode (like any
    ``nn.Module``) regardless of ``decoder``'s own mode — call
    ``.train()``/``.eval()`` on ``deq`` itself to control which path runs::

        deq = QDEQ()
        deq.train()  # or deq.eval() for inference-only, no backward wiring
        z_out, info = deq(decoder, z0, u, pos_emb)
    """

    def __init__(
        self,
        f_solver: Solver | None = None,
        b_solver: Solver | None = None,
        f_thres: int | None = None,
        b_thres: int | None = None,
        stop_mode: str = "rel",
        solve_ctx: Callable[[], ContextManager] = contextlib.nullcontext,
        reattach_ctx: Callable[[], ContextManager] = contextlib.nullcontext,
        *,
        f_eps: float | None = None,
        b_eps: float | None = None,
    ):
        """
        Args:
            f_solver: forward fixed-point solver,
                ``solve(f, z0, threshold, eps=1e-3, stop_mode="rel", **kw) -> dict``.
                Defaults to the Ising/SA solver (``v5_dynamic`` variant) when
                not given.
            b_solver: backward (implicit-gradient) solver; defaults to
                ``anderson_solver`` (NOT to ``f_solver``).
            f_thres: forward solver iteration budget. ``None`` (default) omits
                the ``threshold`` kwarg entirely, so a solver that carries its
                own tuned default (e.g. ``ising_sa_solver``'s
                ``IsingSolverConfig.threshold``) keeps it — passing an int
                here always overrides that default.
            b_thres: same, for the backward solve.
            f_eps: forward tolerance override; None preserves the solver default.
            b_eps: backward tolerance override; None preserves the solver default.
            stop_mode: convergence criterion passed to both solvers.
            solve_ctx: context-manager factory bracketing the no-grad forward solve.
            reattach_ctx: context-manager factory bracketing the single
                differentiable reattach call.
        """
        super().__init__()
        self.f_solver = f_solver if f_solver is not None else _default_ising_solver()
        self.b_solver = b_solver if b_solver is not None else anderson_solver
        self.f_thres = f_thres
        self.b_thres = b_thres
        self.stop_mode = stop_mode
        self.solve_ctx = solve_ctx
        self.reattach_ctx = reattach_ctx
        self.f_eps = f_eps
        self.b_eps = b_eps

    def forward(self, func: Callable[..., torch.Tensor], z_init: torch.Tensor, *func_args):
        """Solve for the fixed point of ``func`` and, in training mode, attach
        the implicit-function-theorem backward pass.

        Returns:
            ``(z_out, info)``. In eval mode, ``z_out`` is the solved fixed
            point with ``requires_grad=False`` and ``info`` is the forward
            solver's result dict, never mutated afterwards. In training mode,
            ``z_out`` requires grad; ``info`` additionally contains
            ``"z_star"`` and ``"new_z"`` (the detached-and-reattached input
            and the one differentiable ``func`` call's output — pass
            ``(info["new_z"], info["z_star"])`` to ``jac_loss_estimate``/
            ``power_method`` above for jacobian regularization or a
            spectral-radius estimate; ``z_out`` itself is *not* the right
            tensor for this — its gradient w.r.t. ``z_star`` runs through the
            implicit-corrected backward solve, not the one-step Jacobian
            these diagnostics estimate. Call them *before* ``.backward()``,
            and fold their result into the loss you then call
            ``.backward()`` on — ``new_z``'s graph is freed by that call
            (PyTorch's normal behavior), so using it afterwards raises
            ``RuntimeError: Trying to backward through the graph a second
            time`` — and ``info`` is *mutated in place* with
            ``backward_<key>`` entries once the caller later calls
            ``.backward()`` on a loss depending on ``z_out`` — read those
            only after backward has run.
        """
        quant_target = _quant_target(func)

        with self.solve_ctx():
            if self.training and quant_target is not None:
                quant_target.set_quant(False)
            try:
                with torch.no_grad():
                    fwd_info = self.f_solver(
                        lambda z: func(z, *func_args),
                        z_init,
                        **_solver_kwargs(self.f_thres, self.stop_mode, self.f_eps),
                    )
                    z_star = fwd_info["result"]
            except BaseException:
                if self.training and quant_target is not None:
                    quant_target.set_quant(True)
                raise

        if not self.training:
            return z_star, fwd_info

        return self._forward_train(func, func_args, z_star, fwd_info, quant_target)

    def _forward_train(self, func, func_args, z_star, fwd_info, quant_target):
        with self.reattach_ctx():
            if quant_target is not None:
                quant_target.set_quant(True)
            z_star = z_star.detach().requires_grad_()
            with torch.enable_grad():
                new_z = func(z_star, *func_args)

        info = dict(fwd_info)
        info["z_star"] = z_star
        info["new_z"] = new_z
        z_out = _DEQImplicitGrad.apply(
            new_z, z_star, self.b_solver, self.b_thres, self.stop_mode, info, self.b_eps
        )
        return z_out, info


__all__ = [
    "QDEQ",
    "IsingSolverConfig",
    "anderson_solver",
    "ising_sa_solver",
    "jac_loss_estimate",
    "power_method",
]
