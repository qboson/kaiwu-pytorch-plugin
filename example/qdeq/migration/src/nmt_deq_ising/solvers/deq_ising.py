"""Ising/QUBO fixed-point solvers for DEQ-style matching decoders."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


@dataclass
class IsingSolverConfig:
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
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in valid})


class LocalAnnealingBackend:
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


class TorchSimulatedAnnealing:
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
        spin = torch.randint(0, 2, (chains, n), device=device, generator=generator, dtype=torch.int64).to(dtype)
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
                accept = (delta <= 0.0) | (torch.rand(chains, device=device, generator=generator) < accept_prob)
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


class CoefficientIsingFixedPointSolver:
    """Solve z = f(z) by Ising-encoded low-dimensional coefficient updates.

    At each outer iteration we build a small basis V, approximate the residual
    change r(z + Va) with finite differences, and choose binary-encoded
    coefficients by minimizing a QUBO/Ising objective.
    """

    def __init__(self, config: IsingSolverConfig):
        self.config = config
        self.backend = LocalAnnealingBackend(config)
        self.torch_backend = TorchSimulatedAnnealing(config)
        self._last_best_energy = float("nan")
        self._last_predicted_energy = float("nan")
        self._last_coeff_norm = float("nan")

    @torch.no_grad()
    def solve(self, f: Callable[[torch.Tensor], torch.Tensor], z0: torch.Tensor) -> dict[str, object]:
        z = z0.detach().clone()
        residual_history: list[torch.Tensor] = []
        update_history: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = []
        abs_trace: list[float] = []
        rel_trace: list[float] = []
        rel_trace_v5: list[float] = []
        energy_trace: list[float] = []
        coeff_norm_trace: list[float] = []
        basis_dim_trace: list[int] = []
        sa_time = 0.0
        start = time.perf_counter()
        best_z = z.clone()
        best_rel = float("inf")

        for step in range(1, self.config.threshold + 1):
            residual = f(z) - z
            abs_res = float(residual.norm().detach().cpu())
            rel_res = float((residual.norm() / z.norm().clamp_min(1e-8)).detach().cpu())
            fz = residual + z
            rel_res_v5 = float((residual.norm() / fz.norm().clamp_min(1e-9)).detach().cpu())
            abs_trace.append(abs_res)
            rel_trace.append(rel_res)
            rel_trace_v5.append(rel_res_v5)
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
            basis_dim_trace.append(len(basis))
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
            energy_trace.append(float(self._last_predicted_energy))
            coeff_norm_trace.append(float(self._last_coeff_norm))
            sa_time += self._last_sa_time

        result = best_z
        final_residual = f(result) - result
        final_abs = final_residual.norm()
        final_fz = final_residual + result
        old_final_rel_debug = final_abs / result.norm().clamp_min(1e-8)
        final_rel_v5 = final_abs / final_fz.norm().clamp_min(1e-9)
        final_rel = final_rel_v5 if self.config.solver_variant == "v5_dynamic" else old_final_rel_debug
        return {
            "result": result,
            "abs_trace": abs_trace,
            "rel_trace": rel_trace,
            "rel_trace_v5": rel_trace_v5,
            "residual_abs_trace": abs_trace,
            "residual_rel_trace_v5": rel_trace_v5,
            "energy_trace": energy_trace,
            "coeff_norm_trace": coeff_norm_trace,
            "basis_dim_trace": basis_dim_trace,
            "nstep": len(abs_trace),
            "final_abs": final_abs,
            "final_rel": final_rel,
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

    def _solve_coefficients(self, residual: torch.Tensor, jv: list[torch.Tensor], step: int) -> np.ndarray:
        if self.config.solver_variant == "v5_dynamic":
            return self._solve_coefficients_v5(residual, jv, step)
        if self.config.solver_backend == "gpu_torch":
            return self._solve_coefficients_torch(residual, jv, step)
        if self.config.solver_backend != "cpu_numpy":
            raise ValueError(f"Unsupported solver_backend: {self.config.solver_backend}")
        basis_count = len(jv)
        bits = self.config.coeff_bits
        weights = np.array([self.config.coeff_step * (2**idx) for idx in range(bits)], dtype=np.float64)
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
        self._last_predicted_energy = float(0.5 * coeff @ quad @ coeff + lin @ coeff)
        self._last_coeff_norm = float(np.linalg.norm(coeff))
        return coeff

    def _solve_coefficients_v5(self, residual: torch.Tensor, jv: list[torch.Tensor], step: int) -> np.ndarray:
        if self.config.solver_backend != "cpu_numpy":
            raise ValueError("v5_dynamic currently supports only solver_backend=cpu_numpy")
        m = len(jv)
        bits = int(self.config.coeff_bits)
        r_flat = residual.reshape(-1)
        jv_flat = torch.stack([v.reshape(-1) for v in jv], dim=0)
        a_mat = 0.5 * (jv_flat @ jv_flat.t()).detach().cpu().double().numpy()
        b_vec = (jv_flat @ r_flat).detach().cpu().double().numpy()
        coeff_min, coeff_step, scale = self._v5_residual_scaled_coeff_spec(float(residual.norm().detach().cpu()), m, bits)
        offset, enc = self._v5_basis_coeff_matrix(m, coeff_min, coeff_step, bits, scale)
        matrix, n_bits = self._v5_build_matrix(a_mat, b_vec, offset, enc)
        spin, elapsed, best_energy = self.backend.solve(matrix, seed=self.config.seed + step)
        self._last_sa_time = elapsed
        self._last_best_energy = best_energy
        pred_energy, coeff = self._v5_candidate_binary_coeffs(spin, n_bits, offset, enc, a_mat, b_vec)
        self._last_predicted_energy = pred_energy
        self._last_coeff_norm = float(np.linalg.norm(coeff))
        return coeff

    @staticmethod
    def _v5_residual_scaled_coeff_spec(r_norm: float, m: int, bits: int) -> tuple[float, float, float]:
        scale = r_norm / np.sqrt(max(1, m))
        return -2.0, 4.0 / ((2**bits) - 1), scale

    @staticmethod
    def _v5_basis_coeff_matrix(m: int, coeff_min: float, coeff_step: float, bits: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
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

    def _v5_build_matrix(self, a_mat: np.ndarray, b_vec: np.ndarray, offset: np.ndarray, enc: np.ndarray) -> tuple[np.ndarray, int]:
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

    def _solve_coefficients_torch(self, residual: torch.Tensor, jv: list[torch.Tensor], step: int) -> list[float]:
        basis_count = len(jv)
        bits = int(self.config.coeff_bits)
        device = residual.device
        dtype = torch.float64
        weights = torch.tensor([self.config.coeff_step * (2**idx) for idx in range(bits)], device=device, dtype=dtype)
        offsets = torch.full((basis_count,), float(self.config.coeff_min), device=device, dtype=dtype)
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
                        q[idx_i, idx_j] = q[idx_i, idx_j] + 0.5 * quad[i, j] * weights[bi] * weights[bj]
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
        coeff_cpu = coeff.detach().cpu().double().numpy()
        quad_cpu = quad.detach().cpu().double().numpy()
        lin_cpu = lin.detach().cpu().double().numpy()
        self._last_predicted_energy = float(0.5 * coeff_cpu @ quad_cpu @ coeff_cpu + lin_cpu @ coeff_cpu)
        self._last_coeff_norm = float(np.linalg.norm(coeff_cpu))
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


class DEQIsingImplicitSolver(nn.Module):
    """DEQ wrapper using Ising in both forward and implicit backward solves."""

    def __init__(self, config: object):
        super().__init__()
        self.config = IsingSolverConfig.from_dict(config)
        self.forward_solver = CoefficientIsingFixedPointSolver(self.config)
        self.backward_solver = CoefficientIsingFixedPointSolver(self.config)
        self.backward_mode = str(config.get("backward", "ising")) if isinstance(config, dict) else "ising"

    def forward(self, f: Callable[[torch.Tensor], torch.Tensor], z0: torch.Tensor) -> tuple[torch.Tensor, dict[str, object]]:
        info = self.forward_solver.solve(f, z0)
        z_star = info["result"]
        if not self.training:
            return z_star, info

        z_req = z_star.detach().requires_grad_(True)
        with torch.enable_grad():
            f_z = f(z_req)
        if self.config.reattach_mode == "deq_sequence":
            z_out = f_z
            info["reattach_mode"] = "deq_sequence"
            info["hook_tensor"] = "f_z"
        elif self.config.reattach_mode == "straight_through":
            z_out = z_star + (f_z - f_z.detach())
            info["reattach_mode"] = "straight_through"
            info["hook_tensor"] = "straight_through_z_star_plus_f_minus_detach"
        else:
            raise ValueError(f"Unsupported reattach_mode: {self.config.reattach_mode}")

        if self.backward_mode == "ising":
            hook_handle = None

            def hook(grad: torch.Tensor) -> torch.Tensor:
                nonlocal hook_handle
                if hook_handle is not None:
                    hook_handle.remove()
                    hook_handle = None

                def backward_fixed_point(y: torch.Tensor) -> torch.Tensor:
                    with torch.enable_grad():
                        jty = torch.autograd.grad(f_z, z_req, y, retain_graph=True, allow_unused=False)[0]
                    return jty + grad

                backward_info = self.backward_solver.solve(backward_fixed_point, torch.zeros_like(grad))
                info["backward_nstep"] = backward_info["nstep"]
                info["backward_final_rel"] = backward_info["final_rel"]
                info["backward_final_rel_v5"] = backward_info.get("final_rel_v5")
                info["backward_old_final_rel_debug"] = backward_info.get("old_final_rel_debug")
                info["backward_energy_last"] = backward_info.get("energy_last")
                info["backward_energy_trace"] = backward_info.get("energy_trace")
                info["backward_rel_trace_v5"] = backward_info.get("rel_trace_v5")
                info["backward_coeff_norm_trace"] = backward_info.get("coeff_norm_trace")
                info["backward_solver_variant"] = backward_info.get("solver_variant")
                info["backward_solver_time"] = backward_info["solver_time"]
                return backward_info["result"]

            hook_handle = z_out.register_hook(hook)
        return z_out, info
