# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Quantum-assisted Deep Equilibrium (QDEQ) layer.

A deep equilibrium layer replaces a stack of weight-tied layers with the
fixed point ``z* = layer(z*, x)`` of a single layer. ``QDEQ`` finds that fixed
point without autograd and computes gradients with the implicit function
theorem, so memory does not grow with the number of solver steps.

Each forward fixed-point update is chosen by solving a small Ising problem
with a Kaiwu SDK optimizer (simulated annealing or CIM), supplied by the
caller in the same way as for ``QVAE`` and ``EnergyModel``. The backward pass
solves the linear implicit-gradient equation with Anderson acceleration.

The backward Anderson solver and the Jacobian regulariser follow the DEQ
reference implementation (https://github.com/locuslab/deq), Copyright (c)
2019 CMU Locus Lab, released under the MIT License.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from ._kaiwu_compat import qubo_matrix_to_ising_matrix

from .usage_stats import kpp_caller_context

__all__ = ["QDEQ", "QDEQConfig"]


@dataclass
class QDEQConfig:
    """Configuration for ``QDEQ``.

    Attributes:

        f_thres: Maximum number of forward solver steps.

        b_thres: Maximum number of backward solver steps.

        f_eps: Forward stopping tolerance.

        b_eps: Backward stopping tolerance.

        stop_mode: ``"rel"`` tracks ``||f(z) - z|| / ||f(z)||``;
            ``"abs"`` tracks ``||f(z) - z||``.

        num_basis: Search directions per Ising step.

        coeff_bits: Binary variables encoding each direction's coefficient.

        coeff_range: Half-width of each coefficient's range, in units of
            ``||f(z) - z|| / sqrt(num_basis)``.

        fd_eps: Step of the finite-difference Jacobian estimate.

        history_size: Past residuals and updates kept as search directions.

        anderson_m: Number of past iterates mixed by the backward Anderson solver.

        anderson_lam: Regularisation of the backward Anderson mixing problem.

        jac_loss_vecs: Probe vectors for the Jacobian regulariser stored in
            ``QDEQ.last_jac_loss``; ``0`` disables it.
    """

    f_thres: int = 12
    b_thres: int = 25
    f_eps: float = 1e-3
    b_eps: float = 1e-3
    stop_mode: str = "rel"
    num_basis: int = 8
    coeff_bits: int = 8
    coeff_range: float = 2.0
    fd_eps: float = 1e-3
    history_size: int = 4
    anderson_m: int = 6
    anderson_lam: float = 1e-4
    jac_loss_vecs: int = 0


def _residual_norms(residual: torch.Tensor, f_value: torch.Tensor) -> tuple[float, float]:
    """Returns the absolute and relative norms of a fixed-point residual.

    Args:
        residual: ``f(z) - z``.
        f_value: ``f(z)``, the denominator of the relative norm.

    Returns:
        tuple: ``||residual||`` and ``||residual|| / ||f_value||``.
    """
    abs_res = residual.norm().item()
    return abs_res, abs_res / (f_value.norm().item() + 1e-9)


def _anderson_solver(
    func: Callable[[torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    config: QDEQConfig,
    threshold: int,
    eps: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Finds ``z = func(z)`` by Anderson-accelerated iteration.

    Args:
        func: Fixed-point map; takes and returns tensors shaped like ``z0``.
        z0: Initial guess with the batch axis first.
        config: Supplies ``stop_mode``, ``anderson_m`` and ``anderson_lam``.
        threshold: Maximum number of iterations.
        eps: Stopping tolerance.

    Returns:
        tuple: The iterate with the lowest tracked residual, and a dict with
        ``"nstep"`` and ``"residual"``.
    """
    # Ring buffers of the last m iterates x_i and their images f(x_i), one row
    # per sample; each sample is mixed independently.
    bsz, m = z0.shape[0], max(config.anderson_m, 2)
    x_hist = torch.zeros(bsz, m, z0[0].numel(), dtype=z0.dtype, device=z0.device)
    f_hist = torch.zeros_like(x_hist)
    # Two plain fixed-point steps seed the history.
    x_hist[:, 0] = z0.reshape(bsz, -1)
    f_hist[:, 0] = func(z0).reshape(bsz, -1)
    x_hist[:, 1] = f_hist[:, 0]
    f_hist[:, 1] = func(f_hist[:, 0].reshape_as(z0)).reshape(bsz, -1)

    # Bordered system whose solution gives mixing weights summing to one.
    h_mat = torch.zeros(bsz, m + 1, m + 1, dtype=z0.dtype, device=z0.device)
    h_mat[:, 0, 1:] = h_mat[:, 1:, 0] = 1
    rhs = torch.zeros(bsz, m + 1, 1, dtype=z0.dtype, device=z0.device)
    rhs[:, 0] = 1

    best_z, best_res, nstep = z0, float("inf"), 0
    for k in range(2, max(threshold, 3)):
        n = min(k, m)
        # alpha = argmin ||sum_i alpha_i (f(x_i) - x_i)||^2  s.t.  sum_i alpha_i = 1.
        g_mat = f_hist[:, :n] - x_hist[:, :n]
        eye = torch.eye(n, dtype=z0.dtype, device=z0.device)[None]
        h_mat[:, 1 : n + 1, 1 : n + 1] = g_mat @ g_mat.transpose(1, 2) + config.anderson_lam * eye
        alpha = torch.linalg.solve(h_mat[:, : n + 1, : n + 1], rhs[:, : n + 1])[:, 1 : n + 1, 0]
        # The next iterate mixes the stored images and overwrites the oldest slot.
        x_hist[:, k % m] = (alpha[:, None] @ f_hist[:, :n])[:, 0]
        f_hist[:, k % m] = func(x_hist[:, k % m].reshape_as(z0)).reshape(bsz, -1)

        nstep = k - 1
        abs_res, rel_res = _residual_norms(f_hist[:, k % m] - x_hist[:, k % m], f_hist[:, k % m])
        tracked = rel_res if config.stop_mode == "rel" else abs_res
        # Iterations are not monotone, so the best iterate seen is returned.
        if tracked < best_res:
            best_res, best_z = tracked, x_hist[:, k % m].reshape_as(z0).clone()
        if tracked < eps:
            break
    return best_z, {"nstep": nstep, "residual": best_res}


def _ising_fixed_point_solver(
    func: Callable[[torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    sampler: Any,
    config: QDEQConfig,
    threshold: int,
    eps: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Finds ``z = func(z)`` with update coefficients chosen by an Ising sampler.

    Each step forms the residual ``r = func(z) - z``, takes ``num_basis``
    directions ``V`` from ``-r`` and recent history, and estimates ``G = J V``
    (``J`` the Jacobian of the residual) by finite differences. The step
    ``z + V a`` minimising ``||r + G a||^2 / 2`` is a quadratic problem in
    ``a``; each coefficient is encoded by ``coeff_bits`` binary variables and
    the resulting QUBO is solved by ``sampler``.

    Args:
        func: Fixed-point map; takes and returns tensors shaped like ``z0``.
        z0: Initial guess.
        sampler: Kaiwu SDK optimizer; see ``QDEQ``.
        config: Supplies the Ising and stopping options.
        threshold: Maximum number of updates.
        eps: Stopping tolerance.

    Returns:
        tuple: The iterate with the lowest tracked residual, and a dict with
        ``"nstep"``, ``"residual"`` and ``"sampler_time"`` (seconds).
    """
    z, residual = z0, func(z0) - z0
    history_r, history_dz, history_dr = [], [], []
    best_z, best_res, nstep, sampler_time = z, float("inf"), 0, 0.0

    for step in range(threshold + 1):
        # Check the current iterate before deciding whether to take another step.
        abs_res, rel_res = _residual_norms(residual, residual + z)
        if not math.isfinite(abs_res):
            break
        tracked = rel_res if config.stop_mode == "rel" else abs_res
        if tracked < best_res:
            best_res, best_z = tracked, z
        if tracked < eps or step == threshold:
            break

        # G[i] = (r(z + h v_i) - r(z)) / h approximates J v_i for each direction v_i.
        basis = _search_directions(residual, history_r, history_dz, history_dr, config.num_basis)
        shifted = z + config.fd_eps * basis
        g_mat = torch.stack([(func(zs) - zs - residual).reshape(-1) for zs in shifted])
        g_mat = (g_mat / config.fd_eps).double()
        start = time.perf_counter()
        coeff = _sample_coefficients(g_mat, residual.reshape(-1).double(), abs_res, sampler, config)
        sampler_time += time.perf_counter() - start
        if coeff is None:
            break

        # z <- z + sum_i a_i v_i, then record the step for the next search directions.
        coeff = torch.as_tensor(coeff, dtype=z.dtype, device=z.device)
        delta = (coeff.view(-1, *[1] * z.ndim) * basis).sum(0)
        next_residual = func(z + delta) - (z + delta)
        history_r = (history_r + [residual])[-config.history_size :]
        history_dz = (history_dz + [delta])[-config.history_size :]
        history_dr = (history_dr + [next_residual - residual])[-config.history_size :]
        z, residual, nstep = z + delta, next_residual, step + 1

    return best_z, {"nstep": nstep, "residual": best_res, "sampler_time": sampler_time}


def _search_directions(residual, history_r, history_dz, history_dr, num_basis) -> torch.Tensor:
    """Builds the unit search directions of one Ising step.

    Directions are taken in order from ``-r``, past residuals and pairs of
    past updates and residual changes, newest first. Missing directions in
    the first steps are filled with random sign vectors.

    Args:
        residual: Current residual ``r = f(z) - z``.
        history_r: Past residuals, oldest first.
        history_dz: Past updates ``z_{k+1} - z_k``, oldest first.
        history_dr: Past residual changes ``r_{k+1} - r_k``, oldest first.
        num_basis: Number of directions to return.

    Returns:
        torch.Tensor: Directions of shape ``(num_basis, *residual.shape)``,
        each normalised to unit norm.
    """
    directions = [-residual, *reversed(history_r)]
    for dz, dr in zip(reversed(history_dz), reversed(history_dr)):
        directions += [dz, dr]
    directions = directions[:num_basis]
    while len(directions) < num_basis:
        directions.append(torch.randint_like(residual, 0, 2) * 2 - 1)
    return torch.stack([v / (v.norm() + 1e-9) for v in directions])


def _sample_coefficients(g_mat, residual, residual_norm, sampler, config):
    """Chooses the step coefficients ``a`` minimising ``||r + G^T a||^2 / 2`` with the sampler.

    The objective is the linearised residual after the step ``z <- z + V^T a``:

    - ``r = f(z) - z`` is the current residual, flattened to length ``n``;
      the fixed point is where ``r = 0``.
    - ``V`` holds the ``num_basis`` search directions ``v_i`` as rows.
    - ``G`` holds the directional derivatives of the residual as rows,
      ``G[i] = J v_i ~ (r(z + h v_i) - r(z)) / h`` with ``J = df/dz - I``
      the Jacobian of ``r`` and ``h = fd_eps``.

    By first-order Taylor expansion ``r(z + V^T a) ~ r + G^T a``, so
    minimising ``||r + G^T a||^2 / 2`` picks the step that best cancels the
    residual. Expanding gives the quadratic form ``a^T H a / 2 + g^T a + const``
    with ``H = G G^T`` and ``g = G r``.

    Each coefficient is encoded by ``config.coeff_bits`` binary variables on a
    uniform grid over ``[-s, s]``, ``s = coeff_range * residual_norm /
    sqrt(num_basis)``. The QUBO is converted to an Ising matrix with
    the installed SDK's QUBO-to-Ising helper and passed to
    ``sampler.solve``; every returned row is decoded and the best one kept.

    Args:
        g_mat: Directional derivatives ``G`` of shape ``(num_basis, n)``.
        residual: Flattened residual ``r`` of length ``n``.
        residual_norm: ``||r||``, which scales the coefficient range.
        sampler: Kaiwu SDK optimizer; see ``QDEQ``.
        config: Supplies ``num_basis``, ``coeff_bits`` and ``coeff_range``.

    Returns:
        np.ndarray: Coefficients of length ``num_basis``, or ``None`` if
        ``G`` or ``r`` contains non-finite values.

    Raises:
        RuntimeError: If ``sampler.solve`` does not return spin rows of the
            Ising matrix size.
    """
    # ||r + G^T a||^2 / 2 = a^T H a / 2 + g^T a + const with H = G G^T, g = G r.
    hess = (g_mat @ g_mat.T).cpu().numpy()
    grad = (g_mat @ residual).cpu().numpy()
    if not (np.all(np.isfinite(hess)) and np.all(np.isfinite(grad))):
        return None

    # a = offset + enc @ x with x binary, spanning [-R, R] * scale in 2**bits levels.
    scale = config.coeff_range * residual_norm / math.sqrt(config.num_basis)
    offset = np.full(config.num_basis, -scale)
    weights = 2.0 * scale / (2**config.coeff_bits - 1) * 2.0 ** np.arange(config.coeff_bits)
    enc = np.kron(np.eye(config.num_basis), weights)

    # min_a a^T H a / 2 + g^T a  ==  min_x x^T Q x  with
    # Q = E^T H E / 2 + diag(E^T (H offset + g)), since x_i^2 = x_i for binary x.
    qubo_mat = enc.T @ hess @ enc / 2 + np.diag(enc.T @ (hess @ offset + grad))
    # Minimising x^T Q x is maximising s^T J s, which is what Kaiwu optimizers do.
    ising_mat, _ = qubo_matrix_to_ising_matrix(qubo_mat)
    ising_mat = ising_mat / max(float(np.abs(ising_mat).max()), 1e-12)

    with kpp_caller_context():
        solution = sampler.solve(ising_mat)
    spins = np.atleast_2d(np.asarray(solution, dtype=np.float64))
    if spins.shape[0] == 0 or spins.shape[1] != enc.shape[1] + 1:
        raise RuntimeError(
            f"sampler must return spin rows of length {enc.shape[1] + 1}, "
            f"got shape {spins.shape}"
        )
    # The last spin is the auxiliary spin; x_i = (s_i s_aux + 1) / 2 is invariant
    # to flipping every spin. Every returned row is decoded and the best kept.
    coeffs = offset + ((spins[:, :-1] * spins[:, -1:] + 1) / 2) @ enc.T
    objective = 0.5 * np.einsum("bi,ij,bj->b", coeffs, hess, coeffs) + coeffs @ grad
    return coeffs[int(np.argmin(objective))]


class QDEQ(nn.Module):
    """Deep equilibrium wrapper that turns a weight-tied layer into a fixed-point layer.

    ``layer(z, x)`` must return a tensor shaped like ``z``. ``QDEQ(x)`` finds
    ``z*`` with ``layer(z*, x) = z*`` and returns ``layer(z*, x)``. In training
    mode gradients reach the layer's parameters and ``x`` through the implicit
    function theorem; the solver steps are never backpropagated through.

    ``layer`` must behave identically while solving and during the final
    call. Subclasses can override ``solve_context`` and ``reattach_context``
    to switch state such as dropout or fake quantisation between the two.

    Args:
        layer: Weight-tied layer ``layer(z, x) -> z_next``.

        sampler: Kaiwu SDK optimizer that chooses each forward update, e.g.
            ``SimulatedAnnealingOptimizer`` or a CIM optimizer.
            ``sampler.solve(ising_mat)`` must return spin rows maximising
            ``s^T ising_mat s``. Subclasses that override
            ``solve_fixed_point`` may pass ``None``.

        config: Solver configuration. Defaults to ``QDEQConfig()``.

    Raises:
        ValueError: If ``config.stop_mode`` is not ``"rel"`` or ``"abs"``.
    """

    def __init__(
        self,
        layer: nn.Module,
        sampler: Any,
        config: QDEQConfig | None = None,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.sampler = sampler
        self.config = config or QDEQConfig()
        if self.config.stop_mode not in ("rel", "abs"):
            raise ValueError("stop_mode must be 'rel' or 'abs'")
        self.last_jac_loss = None
        self._last_stats: dict[str, float] = {}

    def forward(self, x: torch.Tensor, z0: torch.Tensor | None = None) -> torch.Tensor:
        """Returns the layer output at its fixed point.

        Args:
            x: Input injected into every layer call.
            z0: Initial guess for the fixed point. Defaults to zeros shaped like ``x``.

        Returns:
            torch.Tensor: ``layer(z*, x)``.
        """
        z0 = torch.zeros_like(x) if z0 is None else z0
        cfg = self.config
        def func(z):
            return self.layer(z, x)

        # 1. Solve for z* without building an autograd graph.
        with self.solve_context(), torch.no_grad():
            z_star, stats = self.solve_fixed_point(func, z0.detach())
        self._last_stats = {f"forward_{key}": value for key, value in stats.items()}
        self.last_jac_loss = None

        # 2. One differentiable layer call at z* connects the output to the
        # layer's parameters and to x.
        with self.reattach_context():
            if not (self.training and torch.is_grad_enabled()):
                return self.layer(z_star, x)
            z_star = z_star.detach().requires_grad_()
            new_z = self.layer(z_star, x)
        if cfg.jac_loss_vecs > 0:
            self.last_jac_loss = self.jacobian_loss(new_z, z_star, cfg.jac_loss_vecs)
        # 3. Replace the one-step gradient with the implicit-function gradient.
        self._attach_implicit_backward(new_z, z_star)
        return new_z

    def _attach_implicit_backward(self, new_z: torch.Tensor, z_star: torch.Tensor) -> None:
        """Replaces the gradient ``g`` reaching ``new_z`` by the solution of ``y = J^T y + g``.

        ``J = d new_z / d z_star``; this is the implicit function theorem
        gradient of Bai et al. (2019), Theorem 1. The linear fixed point is
        solved by Anderson acceleration with ``b_thres`` and ``b_eps``.

        Args:
            new_z: Layer output at the fixed point.
            z_star: Fixed point with ``requires_grad`` set.
        """
        cfg = self.config

        def backward_hook(grad):
            # The solver's vector-Jacobian products start at new_z again.
            handle.remove()

            # y -> J^T y + g, evaluated with one vector-Jacobian product per call.
            def backward_map(y):
                vjp = torch.autograd.grad(new_z, z_star, y, retain_graph=True, allow_unused=True)[0]
                return grad if vjp is None else vjp + grad

            solution, stats = _anderson_solver(
                backward_map, torch.zeros_like(grad), cfg, cfg.b_thres, cfg.b_eps
            )
            self._last_stats.update({f"backward_{key}": value for key, value in stats.items()})
            # Autograd continues from new_z with the solved gradient.
            return solution

        handle = new_z.register_hook(backward_hook)

    def solve_fixed_point(
        self, func: Callable[[torch.Tensor], torch.Tensor], z0: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Finds ``z = func(z)`` with the Ising solver. Override to use another solver.

        Called under ``torch.no_grad()`` and ``solve_context``.

        Args:
            func: Fixed-point map ``z -> layer(z, x)``.
            z0: Initial guess.

        Returns:
            tuple: The fixed point and a dict of statistics, stored with a
            ``forward_`` prefix in ``get_last_stats()``.

        Raises:
            ValueError: If no sampler was given.
        """
        if self.sampler is None:
            raise ValueError("QDEQ requires a sampler to solve the fixed point")
        cfg = self.config
        return _ising_fixed_point_solver(func, z0, self.sampler, cfg, cfg.f_thres, cfg.f_eps)

    def solve_context(self) -> contextlib.AbstractContextManager:
        """Context entered around the fixed-point solve. Override to switch layer state.

        Returns:
            contextlib.AbstractContextManager: A no-op context by default.
        """
        return contextlib.nullcontext()

    def reattach_context(self) -> contextlib.AbstractContextManager:
        """Context entered around the final layer call. Override to switch layer state.

        Returns:
            contextlib.AbstractContextManager: A no-op context by default.
        """
        return contextlib.nullcontext()

    @staticmethod
    def jacobian_loss(new_z: torch.Tensor, z_star: torch.Tensor, vecs: int = 1) -> torch.Tensor:
        """Hutchinson estimate of ``||d new_z / d z_star||_F^2 / z_star.numel()``.

        Args:
            new_z: Layer output at the fixed point.
            z_star: Fixed point with ``requires_grad`` set.
            vecs: Number of random probe vectors.

        Returns:
            torch.Tensor: Differentiable scalar to add to the training loss.
        """
        # E_p ||J^T p||^2 = ||J||_F^2 for standard normal probes p.
        total = 0
        for _ in range(vecs):
            probe = torch.randn_like(z_star)
            vjp = torch.autograd.grad(new_z, z_star, probe, retain_graph=True, create_graph=True)[0]
            total = total + vjp.norm() ** 2
        return total / vecs / z_star.numel()

    def get_last_stats(self) -> dict[str, float]:
        """Returns solver statistics from the last forward and backward pass.

        Returns:
            dict[str, float]: A copy with ``forward_nstep``, ``forward_residual``
            and ``forward_sampler_time`` (seconds); after
            ``backward()`` also ``backward_nstep`` and ``backward_residual``.
        """
        return dict(self._last_stats)
