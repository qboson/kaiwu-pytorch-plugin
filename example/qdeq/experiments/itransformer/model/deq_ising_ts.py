"""Q-DEQ-Ising-SA wrapper for a single shared iTransformer EncoderLayer.

The Ising/QUBO coefficient solver (``DynamicCoeffIsingSolver``) and the
Anderson implicit-gradient backward solver (``original_anderson``) are ported
unchanged from ``NormMatchTrans/model/deq_ising.py`` and
``NormMatchTrans/model/deq_anderson.py`` -- they only operate on a generic
``func(z) -> z`` callable and a flat tensor, so they are architecture
agnostic. The only new piece here is ``EncoderLayerIsingDEQ``, which treats
one iTransformer ``EncoderLayer`` as the fixed-point map F_theta and injects
the original embedded input at every iteration via LayerNorm(z + x) (iTransformer
uses plain LayerNorm, not the nGPT hypersphere norm the graph-matching project
used, so cosine_norm was swapped for a learnable LayerNorm here).
"""

from __future__ import annotations

import math
import signal
import time
from types import SimpleNamespace
from typing import Callable

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

try:
    import kaiwu
except Exception:  # pragma: no cover - depends on the remote research env.
    kaiwu = None

import os

DEBUG_DUMP_DIR = os.environ.get("QDEQ_DEBUG_DUMP_DIR", "qdeq_debug_dumps")
# Live instrumentation on 2026-07-07 showed kaiwu's SA backend consistently
# returns ~400 candidate rows per call in the *normal*, fast (<0.04s) case --
# the earlier "row explosion" theory was measuring the wrong thing and dumped
# a .npz for nearly every call (14k+ files, 6GB in 12 minutes). Only dump on
# genuinely rare conditions now: a much higher row count, or a slow call.
DEBUG_DUMP_ROW_THRESHOLD = int(os.environ.get("QDEQ_DEBUG_ROW_THRESHOLD", "800"))
DEBUG_DUMP_TIME_THRESHOLD_S = float(os.environ.get("QDEQ_DEBUG_TIME_THRESHOLD_S", "1.0"))


class SASolveTimeout(Exception):
    """Raised when a single kaiwu SA restart exceeds its wall-clock budget.

    Two separate ETTh1 runs stalled for 30+ minutes on what should be a
    sub-second SA call (single-core CPU pegged, GPU near-idle, log not
    advancing). A finite-value guard on the QUBO matrix did not stop the
    second stall, so some legitimate-but-extreme matrix is apparently taking
    the annealer far longer than its configured iteration budget suggests it
    should. Rather than keep chasing the exact input that triggers this in
    the underlying (closed-source) kaiwu backend, every restart call is now
    hard-capped with SIGALRM: a timeout is treated the same as a NaN/Inf
    result by the caller (abandon this DEQ step, keep the last good z).
    """


def _alarm_handler(signum, frame):
    raise SASolveTimeout()


def _run_with_timeout(fn, seconds, *args, **kwargs):
    if not hasattr(signal, "SIGALRM"):
        return fn(*args, **kwargs)  # non-POSIX platform: no-op, run unguarded
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# Anderson solver (ported verbatim from NormMatchTrans/model/deq_anderson.py)
# ---------------------------------------------------------------------------

def original_anderson(
    f: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    *,
    m: int = 6,
    lam: float = 1e-4,
    threshold: int = 30,
    eps: float = 1e-3,
    stop_mode: str = "rel",
    beta: float = 1.0,
    tag: str = "untagged",
) -> dict:
    if stop_mode not in {"rel", "abs"}:
        raise ValueError(f"Unsupported stop_mode={stop_mode!r}")
    start = time.perf_counter()
    time_in_f = 0.0
    time_in_linalg = 0.0
    batch = x0.shape[0]
    flat_dim = x0[0].numel()
    alternative_mode = "rel" if stop_mode == "abs" else "abs"

    x_hist = torch.zeros(batch, m, flat_dim, dtype=x0.dtype, device=x0.device)
    f_hist = torch.zeros_like(x_hist)
    x_hist[:, 0] = x0.reshape(batch, -1)
    _t0 = time.perf_counter()
    f_hist[:, 0] = f(x0).reshape(batch, -1)
    time_in_f += time.perf_counter() - _t0
    x_hist[:, 1] = f_hist[:, 0]
    _t0 = time.perf_counter()
    f_hist[:, 1] = f(f_hist[:, 0].reshape_as(x0)).reshape(batch, -1)
    time_in_f += time.perf_counter() - _t0

    h_mat = torch.zeros(batch, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    h_mat[:, 0, 1:] = 1
    h_mat[:, 1:, 0] = 1
    y = torch.zeros(batch, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    trace = {"abs": [], "rel": []}
    lowest = {"abs": float("inf"), "rel": float("inf")}
    lowest_step = {"abs": 0, "rel": 0}
    lowest_x = x0.detach().clone()

    for k in range(2, threshold):
        n = min(k, m)
        _t0 = time.perf_counter()
        residual_hist = f_hist[:, :n] - x_hist[:, :n]
        h_mat[:, 1 : n + 1, 1 : n + 1] = (
            torch.bmm(residual_hist, residual_hist.transpose(1, 2))
            + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        )
        alpha = torch.linalg.solve(h_mat[:, : n + 1, : n + 1], y[:, : n + 1])[:, 1 : n + 1, 0]
        x_new = beta * (alpha[:, None] @ f_hist[:, :n])[:, 0] + (1.0 - beta) * (
            alpha[:, None] @ x_hist[:, :n]
        )[:, 0]
        time_in_linalg += time.perf_counter() - _t0
        _t0 = time.perf_counter()
        f_new = f(x_new.reshape_as(x0)).reshape(batch, -1)
        time_in_f += time.perf_counter() - _t0
        slot = k % m
        x_hist[:, slot] = x_new
        f_hist[:, slot] = f_new

        residual = (f_new - x_new).view_as(x0)
        abs_diff = float(residual.norm().detach().cpu())
        rel_diff = abs_diff / (float(f_new.norm().detach().cpu()) + 1e-5)
        diff = {"abs": abs_diff, "rel": rel_diff}
        trace["abs"].append(abs_diff)
        trace["rel"].append(rel_diff)

        for mode in ("rel", "abs"):
            if diff[mode] < lowest[mode]:
                if mode == stop_mode:
                    lowest_x = x_new.view_as(x0).detach().clone()
                lowest[mode] = diff[mode]
                lowest_step[mode] = k

        if trace[stop_mode][-1] < eps:
            for _ in range(threshold - 1 - k):
                trace[stop_mode].append(lowest[stop_mode])
                trace[alternative_mode].append(lowest[alternative_mode])
            break

    final_rel = torch.tensor(lowest[stop_mode], dtype=x0.dtype, device=x0.device)
    total_time = time.perf_counter() - start
    print(
        f"[ANDERSON-TIMING] tag={tag} nstep={lowest_step[stop_mode]} final_rel={lowest[stop_mode]:.6f} total={total_time:.4f}s | "
        f"func_eval={time_in_f:.4f}s({100*time_in_f/max(total_time,1e-9):.0f}%) "
        f"linalg_solve={time_in_linalg:.4f}s({100*time_in_linalg/max(total_time,1e-9):.0f}%)",
        flush=True,
    )
    return {
        "result": lowest_x,
        "lowest": lowest[stop_mode],
        "nstep": lowest_step[stop_mode],
        "abs_trace": trace["abs"],
        "rel_trace": trace["rel"],
        "eps": eps,
        "threshold": threshold,
        "final_rel": final_rel,
        "solver_time": total_time,
        "time_in_f": time_in_f,
        "time_in_linalg": time_in_linalg,
    }


# ---------------------------------------------------------------------------
# Ising/QUBO coefficient solver (ported verbatim from
# NormMatchTrans/model/deq_ising.py)
# ---------------------------------------------------------------------------

def jac_loss_estimate(f0: torch.Tensor, z0: torch.Tensor, vecs: int = 1, create_graph: bool = True) -> torch.Tensor:
    """Hutchinson estimator of tr(J^T J) / numel(z0), ported verbatim from
    locuslab/deq's lib/jacobian.py. Used as a training-time regularizer on
    the fixed-point map's Jacobian at z_star, matching the original DEQ
    paper's "Stabilizing Equilibrium Models by Jacobian Regularization"
    (arXiv:2106.14342) -- this is what the reference training script actually
    does to keep the equilibrium map well-conditioned, not a special weight
    initialization scheme."""
    result = 0.0
    for _ in range(vecs):
        v = torch.randn_like(z0)
        vJ = autograd.grad(f0, z0, v, retain_graph=True, create_graph=create_graph)[0]
        result = result + vJ.norm() ** 2
    return result / vecs / z0.numel()


def normalize_basis(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-9)


class KaiwuSASolver:
    def __init__(self, args):
        if kaiwu is None:
            raise RuntimeError("kaiwu is not importable; run in the kaiwu1033_py38 environment")
        if getattr(args, "kaiwu_user_id", None) and getattr(args, "kaiwu_sdk_code", None):
            kaiwu.license.init(args.kaiwu_user_id, args.kaiwu_sdk_code)
        self.args = args

    def _raw_solve(self, matrix, seed):
        opt = kaiwu.classical.SimulatedAnnealingOptimizer(
            initial_temperature=self.args.sa_initial_temperature,
            alpha=self.args.sa_alpha,
            cutoff_temperature=self.args.sa_cutoff_temperature,
            iterations_per_t=self.args.sa_iterations_per_t,
            size_limit=getattr(self.args, "sa_size_limit", 100),
            rand_seed=int(seed),
            process_num=1,
        )
        opt.set_matrix(matrix.astype(np.float64))
        return np.asarray(opt.solve())

    def solve(self, matrix, seed):
        results = []
        timeout_s = float(getattr(self.args, "sa_restart_timeout_s", 10.0))
        for i in range(self.args.sa_restarts):
            results.append(_run_with_timeout(self._raw_solve, timeout_s, -matrix, seed + 1009 * i))
        return np.concatenate(results, axis=0) if len(results) > 1 else results[0]


class KaiwuCIMSolver:
    """Kaiwu FastCIM backend, ported verbatim from NormMatchTrans/model/deq_ising.py.

    Submits real tasks to the remote CIM service under ``cim_project_no`` in
    ``task_mode="quota"``. Unlike KaiwuSASolver this consumes a real, shared
    quota -- only wired up here because the user explicitly confirmed the
    existing project_no=YOUR_CIM_PROJECT credential (from the graph-matching CIM
    config) is authorized for this iTransformer experiment too.
    """

    def __init__(self, args):
        if kaiwu is None:
            raise RuntimeError("kaiwu is not importable; run in the kaiwu1033_py38 environment")
        if getattr(args, "kaiwu_user_id", None) and getattr(args, "kaiwu_sdk_code", None):
            kaiwu.license.init(args.kaiwu_user_id, args.kaiwu_sdk_code)
        self.args = args
        self.project_no = str(getattr(args, "cim_project_no", "YOUR_CIM_PROJECT"))
        self.task_prefix = getattr(args, "cim_task_prefix", "qdeq_ts_cim")
        self.priority = int(getattr(args, "cim_priority", 10))
        self.interval = float(getattr(args, "cim_interval", 0.01))
        self.precision_bits = int(getattr(args, "cim_precision_bits", 14))
        self.truncated_precision = int(getattr(args, "cim_truncated_precision", 18))
        save_dir = getattr(args, "cim_save_dir", "cim_tasks")
        os.makedirs(save_dir, exist_ok=True)
        kaiwu.common.CheckpointManager.save_dir = str(save_dir)

    def solve(self, matrix, seed):
        matrix = np.asarray(matrix, dtype=np.float64)
        matrix = 0.5 * (matrix + matrix.T)
        np.fill_diagonal(matrix, 0.0)

        task_name = f"{self.task_prefix}_{int(seed)}"
        worker = kaiwu.cim.FastCIMOptimizer(
            task_name=task_name,
            task_mode="quota",
            need_verify=False,
            wait=True,
            interval=self.interval,
            project_no=self.project_no,
            priority=self.priority,
        )
        sampler = kaiwu.cim.PrecisionReducer(
            worker,
            precision=self.precision_bits,
            truncated_precision=self.truncated_precision,
            only_feasible_solution=False,
        )

        cim_matrix = (-matrix).astype(np.float64)
        max_retries = int(getattr(self.args, "cim_max_retries", 20))
        for attempt in range(1, max_retries + 1):
            try:
                return np.asarray(sampler.solve(cim_matrix))
            except Exception as exc:
                message = str(exc)
                retryable = (
                    isinstance(exc, requests.exceptions.RequestException)
                    or "Read timed out" in message
                    or "ReadTimeout" in message
                    or "Connection aborted" in message
                    or "Connection reset" in message
                    or "Bad Gateway" in message
                    or "502" in message
                    or "Failed to retrieve task" in message
                    or "temporarily unavailable" in message.lower()
                    or "gateway" in message.lower()
                    or "NameResolutionError" in message
                    or "Failed to resolve" in message
                    or "Max retries exceeded" in message
                )
                if not retryable or attempt >= max_retries:
                    raise
                wait_sec = min(300, 30 * attempt)
                print(
                    f"[CIM retry] task={task_name} attempt={attempt}/{max_retries} "
                    f"after error: {message}; sleep {wait_sec}s",
                    flush=True,
                )
                time.sleep(wait_sec)
        raise RuntimeError(f"CIM task {task_name} failed after retries")


class DynamicCoeffIsingSolver:
    """Faithful port of NormMatchTrans's DynamicCoeffIsingSolver (SA + CIM backends)."""

    def __init__(self, args, variant="residual_scaled"):
        self.args = args
        self.variant = variant
        backend = getattr(args, "ising_backend", "sa")
        if backend == "sa":
            self.backend = KaiwuSASolver(args)
        elif backend == "cim":
            self.backend = KaiwuCIMSolver(args)
        else:
            raise ValueError(f"unknown ising_backend: {backend}")
        self.call_index = 0

    def _make_basis(self, z, r, residual_history=None, update_history=None):
        basis = [normalize_basis(-r)]
        residual_history = residual_history or []
        update_history = update_history or []
        if self.args.basis_mode in ("residual_history", "mixed_history"):
            for rr in reversed(residual_history):
                if len(basis) >= self.args.ising_basis:
                    break
                basis.append(normalize_basis(rr))
        if self.args.basis_mode in ("update_history", "mixed_history"):
            for dz, dr in reversed(update_history):
                if len(basis) < self.args.ising_basis:
                    basis.append(normalize_basis(dz))
                if len(basis) < self.args.ising_basis:
                    basis.append(normalize_basis(dr))
                if len(basis) >= self.args.ising_basis:
                    break
        while len(basis) < self.args.ising_basis:
            rnd = torch.randint_like(z, low=0, high=2).float() * 2 - 1
            basis.append(normalize_basis(rnd))
        return basis[: self.args.ising_basis]

    def _coeff_spec(self, step, r_norm, m, coeff_range_override=None):
        scale = r_norm / math.sqrt(max(1, m))
        bits = self.args.coeff_bits
        if coeff_range_override is not None:
            coeff_range = float(coeff_range_override)
        else:
            coeff_range = float(getattr(self.args, "ising_coeff_range", 2.0))
        if self.variant == "residual_scaled":
            return -coeff_range, (2.0 * coeff_range) / ((2**bits) - 1), bits, scale
        raise ValueError(f"unknown Ising variant: {self.variant}")

    def _basis_coeff_matrix(self, m, coeff_min, coeff_step, bits, scale):
        n = m * bits
        B = np.zeros((m, n), dtype=np.float64)
        offset = np.full(m, coeff_min * scale, dtype=np.float64)
        for i in range(m):
            for j in range(bits):
                B[i, i * bits + j] = scale * coeff_step * (2**j)
        return offset, B

    def _basis_coeff_matrix_centered(self, m, center, halfwidth, bits):
        """Like _basis_coeff_matrix but centered per-dimension on `center`
        (length-m real-valued array from a prior coarse solve) with a
        shared +/-halfwidth window, instead of a uniform scalar coeff_min.
        Used for the two-stage coarse->fine refinement: reuses the SAME
        (A, b) local quadratic model as the coarse pass (no new JVP evals
        needed), just re-parametrizes which real values the n_bits spins
        decode to, so the fine pass can resolve within the coarse cell
        that contains the true (unconstrained) optimum.
        """
        n = m * bits
        B = np.zeros((m, n), dtype=np.float64)
        coeff_step = (2.0 * halfwidth) / ((2**bits) - 1)
        offset = np.asarray(center, dtype=np.float64) - halfwidth
        for i in range(m):
            for j in range(bits):
                B[i, i * bits + j] = coeff_step * (2**j)
        return offset, B

    def _candidate_binary_coeffs(self, result, n_bits, offset, B, A, b):
        arr = np.asarray(result)
        if arr.ndim == 1:
            arr = arr[None, :]
        # Hard cap: this loop was observed (py-spy, live process) stuck for
        # 12+ minutes of active CPU on a single call, which only makes sense
        # if the SA backend returned a pathologically large number of
        # candidate rows for some input. Nothing in this codebase should ever
        # need more than a few thousand candidates evaluated per DEQ step, so
        # bound it unconditionally rather than trusting the backend's output
        # size.
        MAX_CANDIDATE_ROWS = 2000
        if arr.shape[0] > MAX_CANDIDATE_ROWS:
            print(f"[Q-DEQ-Ising] SA backend returned {arr.shape[0]} candidate rows "
                  f"(expected ~sa_restarts={self.args.sa_restarts}); "
                  f"capping to {MAX_CANDIDATE_ROWS}.", flush=True)
            arr = arr[:MAX_CANDIDATE_ROWS]
        candidates = []
        for row in arr:
            if row.shape[0] < n_bits + 1:
                continue
            raw = np.sign(row[: n_bits + 1]).astype(np.float64)
            raw[raw == 0] = 1
            spin = raw[:n_bits] * raw[n_bits]
            x = (spin + 1.0) / 2.0
            coeff = offset + B @ x
            energy = float(coeff @ A @ coeff + b @ coeff)
            candidates.append((energy, coeff))
        if not candidates:
            x = np.zeros(n_bits, dtype=np.float64)
            coeff = offset + B @ x
            candidates.append((float(coeff @ A @ coeff + b @ coeff), coeff))
        candidates.sort(key=lambda item: item[0])
        return candidates[0]

    def _build_matrix(self, A, b, offset, B):
        qmat = B.T @ A @ B
        qlin = 2.0 * (offset @ A @ B) + (b @ B)
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
        return mat / max_abs * self.args.ising_matrix_scale, n_bits

    @torch.no_grad()
    def solve(self, func: Callable[[torch.Tensor], torch.Tensor], z0, threshold, eps=1e-3, stop_mode="rel", tag="untagged"):
        z = z0.detach()
        final_z = z.clone()
        abs_trace, rel_trace = [], []
        residual_history = []
        _r_calib_estimates = []
        _r_locked = None
        update_history = []
        _trust_scale = 1.0

        # Fine-grained timing buckets, requested to settle "is JVP or SA the
        # bottleneck" with real numbers instead of estimates.
        t_residual = 0.0   # cost of residual_at(z) -- base function evaluation
        t_basis = 0.0      # cost of _make_basis (pure tensor bookkeeping, no network)
        t_jvp = 0.0        # cost of the finite-difference JVP loop (ising_basis network evals)
        t_matrix = 0.0     # cost of A/b/offset/B/mat construction (numpy, CPU)
        t_sa = 0.0         # cost of backend.solve (the actual kaiwu/SA call)
        t_candidate = 0.0  # cost of _candidate_binary_coeffs (scoring returned candidates)

        def residual_at(zz):
            return func(zz) - zz

        for step in range(1, threshold + 1):
            _t0 = time.perf_counter()
            r = residual_at(z)
            t_residual += time.perf_counter() - _t0
            fz = r + z
            abs_res = float(r.norm().item())
            rel_res = float(abs_res / (fz.norm().item() + 1e-9))

            # Diverged/NaN guard: a runaway residual can produce a NaN/Inf QUBO
            # matrix, which some SA backends handle by spinning far longer
            # than their configured iteration budget instead of erroring out.
            # Bail out to the last good z rather than ever handing kaiwu a
            # non-finite matrix.
            if not math.isfinite(abs_res) or not math.isfinite(rel_res):
                break

            abs_trace.append(abs_res)
            rel_trace.append(rel_res)
            if (rel_res if stop_mode == "rel" else abs_res) < eps:
                final_z = z.clone()
                break

            _t0 = time.perf_counter()
            basis = self._make_basis(z, r, residual_history, update_history)
            t_basis += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            if getattr(self.args, "ising_secant_probe", False):
                # Secant-style JVP probe: the local fd_eps-based derivative
                # (fd_eps ~ 0.3-1.0) is being used to LINEARLY EXTRAPOLATE out
                # to the actually-applied coefficient, which for hard batches
                # reaches |c|~R*scale with R up to 16-20x scale (measured
                # earlier) -- i.e. the model is evaluated ~10-40x beyond the
                # neighborhood where its derivative was actually measured.
                # Instead of estimating an infinitesimal derivative and
                # extrapolating it, probe at a step size proportional to the
                # coefficient magnitude that will actually be used, so the
                # resulting linear model is a SECANT fit to the real
                # trajectory instead of a tangent extrapolated far past its
                # valid neighborhood.
                _scale_secant = abs_res / math.sqrt(max(1, len(basis)))
                _R_secant = float(getattr(self.args, "ising_coeff_range", 2.0))
                _secant_ratio = float(getattr(self.args, "ising_secant_ratio", 0.5))
                fd_eps = max(_secant_ratio * _R_secant * _scale_secant, 1e-4)
            else:
                fd_eps_ratio = getattr(self.args, "fd_eps_scale_ratio", None)
                if fd_eps_ratio is not None:
                    # Floored scale-relative fd_eps: earlier testing (uniform
                    # session) found fd_eps=ratio*scale with a near-zero floor
                    # (1e-6) collapses into the same noise-dominated regime that
                    # made the fixed fd_eps=0.001 default catastrophic, once
                    # scale shrinks late in the trajectory. Clamping to a floor
                    # in the empirically-good fixed-fd_eps range keeps early/mid
                    # steps behaving like the floor (known good) while still
                    # allowing fd_eps to track scale (and shrink further) only
                    # when scale is large enough that doing so is safe.
                    _scale_now = abs_res / math.sqrt(max(1, len(basis)))
                    _fd_floor = float(getattr(self.args, "fd_eps_scale_floor", 0.3))
                    fd_eps = max(float(fd_eps_ratio) * _scale_now, _fd_floor)
                else:
                    fd_eps = self.args.fd_eps
            if getattr(self.args, "ising_central_diff", False):
                # Central-difference JVP: O(fd_eps^2) truncation error vs the
                # one-sided/forward difference's O(fd_eps) -- a strictly more
                # accurate local-derivative estimate for the SAME fd_eps, at
                # 2x the residual_at() cost per basis direction. Motivated by
                # the predicted-vs-actual gap finding (ETTh2: gap grows ~9x
                # for batches that fail to converge, specifically in the late
                # steps) -- that gap reflects the local LINEAR MODEL itself
                # being wrong, not fd_eps being numerically unsafe (fixed
                # fd_eps in the 0.3-1.0 range was already established safe
                # from noise), so a higher-order-accurate derivative directly
                # attacks the diagnosed mechanism instead of just changing
                # fd_eps's magnitude.
                jvs = [(residual_at(z + fd_eps * v) - residual_at(z - fd_eps * v)) / (2.0 * fd_eps) for v in basis]
            else:
                jvs = [(residual_at(z + fd_eps * v) - r) / fd_eps for v in basis]
            t_jvp += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            m = len(basis)
            r_flat = r.reshape(-1)
            jv_flat = torch.stack([jv.reshape(-1) for jv in jvs], dim=0)
            A = 0.5 * (jv_flat @ jv_flat.t()).detach().cpu().double().numpy()
            b = (jv_flat @ r_flat).detach().cpu().double().numpy()
            if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
                break

            if getattr(self.args, "ising_probe_diag", False):
                # Subspace coverage: the best ANY linear combination of the
                # current m basis directions could do (in the local linear
                # model) is project r onto span(jv_flat). The leftover
                # orthogonal component is a hard floor on achievable residual
                # reduction that NO amount of coeff_range widening can fix --
                # only changing/adding basis directions can. ||r_perp||/||r||
                # computed directly from (A,b) already built this step.
                _r_norm_sq = float((r_flat.double() @ r_flat.double()).item())
                try:
                    _alpha_ls = np.linalg.solve(2.0 * A + 1e-10 * np.eye(A.shape[0]), b)
                    _perp_sq = max(_r_norm_sq - float(b @ _alpha_ls), 0.0)
                except np.linalg.LinAlgError:
                    _perp_sq = float('nan')
                _perp_frac = math.sqrt(_perp_sq / max(_r_norm_sq, 1e-300))
                print('[SUBSPACE-COVERAGE] tag=' + tag + ' step=' + str(step) + ' perp_residual_frac=' + format(_perp_frac, '.5f'), flush=True)
                _scale_p = r.norm().item() / math.sqrt(max(1, m))
                try:
                    _c_p = -0.5 * np.linalg.solve(A + 1e-8 * np.eye(A.shape[0]), b)
                except np.linalg.LinAlgError:
                    _c_p = -0.5 * np.linalg.lstsq(A, b, rcond=None)[0]
                _ratio_p = np.abs(_c_p) / max(_scale_p, 1e-12)
                print('[PROBE] tag=' + tag + ' step=' + str(step) + ' ratio=' + str(_ratio_p.tolist()) + ' max=' + format(float(_ratio_p.max()), '.5f'), flush=True)
                _damp_rel = max(1e-4 * float(np.trace(A)) / A.shape[0], 1e-10)
                try:
                    _c_pd = -0.5 * np.linalg.solve(A + _damp_rel * np.eye(A.shape[0]), b)
                except np.linalg.LinAlgError:
                    _c_pd = -0.5 * np.linalg.lstsq(A, b, rcond=None)[0]
                _ratio_pd = np.abs(_c_pd) / max(_scale_p, 1e-12)
                print('[PROBE-DAMPED] tag=' + tag + ' step=' + str(step) + ' damp=' + format(_damp_rel, '.6g') + ' ratio=' + str(_ratio_pd.tolist()) + ' max=' + format(float(_ratio_pd.max()), '.5f'), flush=True)
                try:
                    _eigs = np.linalg.eigvalsh(A)
                    _eig_min = float(_eigs[0])
                    _eig_max = float(_eigs[-1])
                    _cond = _eig_max / max(abs(_eig_min), 1e-300) if _eig_min > 1e-300 else float('inf')
                    _b_norm = float(np.linalg.norm(b))
                    _r_norm_val = float(r.norm().item())
                    print('[COND] tag=' + tag + ' step=' + str(step) + ' eig_min=' + format(_eig_min, '.6g') + ' eig_max=' + format(_eig_max, '.6g') + ' cond=' + format(_cond, '.6g') + ' b_norm=' + format(_b_norm, '.6g') + ' r_norm=' + format(_r_norm_val, '.6g'), flush=True)
                except np.linalg.LinAlgError:
                    pass
            if getattr(self.args, "ising_dynamic_range", False):
                # Continuously-tracking range: unlike ising_adaptive_range (which
                # calibrates once during the first few steps then locks), this
                # recomputes the local unconstrained optimum c* EVERY step and
                # uses margin*|c*|/scale as THAT step's coeff_range. Motivated by
                # measured data showing |c*|/scale grows monotonically through
                # the solve trajectory for the batches that fail to converge
                # (e.g. weather: 1.7 at step0 -> 5.3 at step40 for capped calls,
                # vs staying ~flat for calls that do converge) -- a fixed R
                # chosen from early-step statistics becomes too small exactly
                # when it matters, late in the trajectory. No extra SA calls;
                # reuses the same (A, b) already built this step.
                _scale_dyn = r.norm().item() / math.sqrt(max(1, m))
                _damp_dyn = max(1e-4 * float(np.trace(A)) / A.shape[0], 1e-10)
                try:
                    _c_dyn = -0.5 * np.linalg.solve(A + _damp_dyn * np.eye(A.shape[0]), b)
                except np.linalg.LinAlgError:
                    _c_dyn = -0.5 * np.linalg.lstsq(A, b, rcond=None)[0]
                _ratio_dyn = float(np.max(np.abs(_c_dyn))) / max(_scale_dyn, 1e-12)
                margin = float(getattr(self.args, "ising_dynamic_margin", 1.5))
                r_min = float(getattr(self.args, "ising_r_min", 0.5))
                r_max = float(getattr(self.args, "ising_r_max", 40.0))
                active_range = min(max(margin * _ratio_dyn, r_min), r_max)
                if getattr(self.args, "ising_probe_diag", False):
                    print('[ISING-DYNAMIC-R] tag=' + tag + ' step=' + str(step) + ' ratio=' + format(_ratio_dyn, '.4f') + ' active_R=' + format(active_range, '.4f'), flush=True)
                coeff_min, coeff_step, bits, scale = self._coeff_spec(step, r.norm().item(), m, coeff_range_override=active_range)
            elif getattr(self.args, "ising_adaptive_range", False):
                # Online R calibration: the first `ising_r_calibration_steps`
                # DEQ-solve steps each probe the unconstrained optimum of the
                # SAME local quadratic model (A, b) we already build every step
                # -- a cheap 8x8 linear solve, no extra SA calls -- and use its
                # magnitude (in units of `scale`) to lock in a coeff_range for
                # the REMAINING steps of this same solve() call. Total step
                # budget and SA-calls-per-step are both unchanged; only the
                # quantization range adapts to the actually-observed local
                # coefficient scale instead of a hand-picked constant.
                _scale_probe = r.norm().item() / math.sqrt(max(1, m))
                try:
                    _c_probe = -0.5 * np.linalg.solve(A + 1e-8 * np.eye(A.shape[0]), b)
                except np.linalg.LinAlgError:
                    _c_probe = -0.5 * np.linalg.lstsq(A, b, rcond=None)[0]
                _r_probe = float(np.max(np.abs(_c_probe))) / max(_scale_probe, 1e-12)
                calib_steps = int(getattr(self.args, "ising_r_calibration_steps", 3))
                margin = float(getattr(self.args, "ising_r_margin", 2.0))
                r_min = float(getattr(self.args, "ising_r_min", 0.5))
                r_max = float(getattr(self.args, "ising_r_max", 40.0))
                if _r_locked is None and len(_r_calib_estimates) < calib_steps:
                    _r_calib_estimates.append(_r_probe)
                    active_range = float(getattr(self.args, "ising_coeff_range", 12.0))
                elif _r_locked is None:
                    _r_locked = min(max(margin * max(_r_calib_estimates), r_min), r_max)
                    active_range = _r_locked
                    print('[ISING-ADAPTIVE-R] tag=' + tag + ' locked_at_step=' + str(step) + ' calib_estimates=' + str(_r_calib_estimates) + ' locked_R=' + format(_r_locked, '.3f'), flush=True)
                else:
                    active_range = _r_locked
                coeff_min, coeff_step, bits, scale = self._coeff_spec(step, r.norm().item(), m, coeff_range_override=active_range)
            else:
                coeff_min, coeff_step, bits, scale = self._coeff_spec(step, r.norm().item(), m)
            offset, B = self._basis_coeff_matrix(m, coeff_min, coeff_step, bits, scale)
            mat, n_bits = self._build_matrix(A, b, offset, B)
            t_matrix += time.perf_counter() - _t0
            if not np.all(np.isfinite(mat)):
                break

            solve_start = time.perf_counter()
            this_call_index = self.call_index
            try:
                result = self.backend.solve(mat, self.call_index + self.args.seed)
            except SASolveTimeout:
                print(
                    f"[Q-DEQ-Ising] SA restart exceeded its timeout at solve step {step} "
                    f"(call_index={self.call_index}, tag={tag}); abandoning this DEQ step, keeping last z.",
                    flush=True,
                )
                break
            finally:
                self.call_index += 1
            solve_elapsed = time.perf_counter() - solve_start
            t_sa += solve_elapsed
            print(f"[CIM-CALL] tag={tag} step={step} call={this_call_index} elapsed={solve_elapsed:.3f}s", flush=True)

            raw_result = np.asarray(result)
            raw_rows = raw_result.shape[0] if raw_result.ndim >= 1 else 1

            _time_threshold = DEBUG_DUMP_TIME_THRESHOLD_S
            if getattr(self.args, "ising_backend", "sa") == "cim":
                _time_threshold = float(os.environ.get("QDEQ_DEBUG_TIME_THRESHOLD_CIM_S", "25.0"))
            if raw_rows > DEBUG_DUMP_ROW_THRESHOLD or solve_elapsed > _time_threshold:
                try:
                    os.makedirs(DEBUG_DUMP_DIR, exist_ok=True)
                    dump_path = os.path.join(
                        DEBUG_DUMP_DIR, f"anomaly_{tag}_step{step}_call{this_call_index}.npz"
                    )
                    np.savez(
                        dump_path,
                        mat=mat,
                        A=A,
                        b=b,
                        offset=offset,
                        B=B,
                        raw_result=raw_result,
                        z=z.detach().cpu().numpy(),
                        r=r.detach().cpu().numpy(),
                        coeff_min=coeff_min,
                        coeff_step=coeff_step,
                        bits=bits,
                        scale=scale,
                        seed=this_call_index + self.args.seed,
                        sa_restarts=self.args.sa_restarts,
                        sa_initial_temperature=self.args.sa_initial_temperature,
                        sa_alpha=self.args.sa_alpha,
                        sa_cutoff_temperature=self.args.sa_cutoff_temperature,
                        sa_iterations_per_t=self.args.sa_iterations_per_t,
                    )
                    print(f"[Q-DEQ-Ising-DIAG] ANOMALY dumped to {dump_path} (raw_rows={raw_rows}, solve_time={solve_elapsed:.2f}s)", flush=True)
                except Exception as dump_exc:
                    print(f"[Q-DEQ-Ising-DIAG] failed to dump anomaly: {dump_exc}", flush=True)

            _t0 = time.perf_counter()
            _, coeff_np = self._candidate_binary_coeffs(result, n_bits, offset, B, A, b)
            t_candidate += time.perf_counter() - _t0
            if not np.all(np.isfinite(coeff_np)):
                break

            if getattr(self.args, "ising_probe_diag", False):
                _boundary = -coeff_min * scale
                _qstep = scale * coeff_step
                _hits = int(np.sum(np.abs(np.abs(coeff_np) - _boundary) < 1.5 * _qstep))
                print('[BOUNDARY-CHECK] tag=' + tag + ' step=' + str(step) + ' hits=' + str(_hits) + '/' + str(len(coeff_np)) + ' boundary=' + format(_boundary, '.5f') + ' coeff=' + str(coeff_np.tolist()), flush=True)

            if getattr(self.args, "ising_two_stage", False):
                # Coarse->fine refinement: reuse the SAME (A, b) local
                # quadratic model (no new JVP evals -- the expensive part),
                # just re-solve a second, much narrower QUBO centered on
                # the coarse candidate to resolve within its quantization
                # cell. Falls back to the coarse coeff_np on any failure.
                halfwidth = 0.5 * coeff_step * scale
                offset2, B2 = self._basis_coeff_matrix_centered(m, coeff_np, halfwidth, bits)
                mat2, n_bits2 = self._build_matrix(A, b, offset2, B2)
                if np.all(np.isfinite(mat2)):
                    _t0 = time.perf_counter()
                    try:
                        result2 = self.backend.solve(mat2, self.call_index + self.args.seed)
                    except SASolveTimeout:
                        result2 = None
                    finally:
                        self.call_index += 1
                    if result2 is not None:
                        t_sa += time.perf_counter() - _t0
                        _, coeff_np2 = self._candidate_binary_coeffs(result2, n_bits2, offset2, B2, A, b)
                        if np.all(np.isfinite(coeff_np2)):
                            coeff_np = coeff_np2

            _predict_diag = getattr(self.args, "ising_probe_diag", False)
            _trust_on = getattr(self.args, "ising_trust_region", False)
            if _predict_diag or _trust_on:
                _coeff_t = torch.tensor(coeff_np, dtype=jv_flat.dtype, device=jv_flat.device)
                _r_pred_flat = r_flat + (_coeff_t @ jv_flat)
                _pred_norm = float(_r_pred_flat.norm().item())
                _r_norm_now = float(r.norm().item())
                _pred_ratio = _pred_norm / max(_r_norm_now, 1e-300)

            coeff = torch.tensor(coeff_np, dtype=z.dtype, device=z.device)
            if _trust_on:
                coeff = coeff * _trust_scale
            delta = sum(coeff[i] * basis[i] for i in range(m))
            z_prev, r_prev = z, r
            z = z + delta
            _t0 = time.perf_counter()
            r_next = residual_at(z)
            t_residual += time.perf_counter() - _t0
            if _predict_diag or _trust_on:
                _actual_ratio = float(r_next.norm().item()) / max(_r_norm_now, 1e-300)
                _gap = _actual_ratio - _pred_ratio
                if _predict_diag:
                    print('[PREDICT-VS-ACTUAL] tag=' + tag + ' step=' + str(step) + ' predicted_ratio=' + format(_pred_ratio, '.5f') + ' actual_ratio=' + format(_actual_ratio, '.5f') + ' gap=' + format(_gap, '.5f') + ' trust_scale=' + format(_trust_scale, '.4f'), flush=True)
                if _trust_on:
                    # Trust-region step damping: when the linear (JVP-based)
                    # model's prediction disagrees badly with what actually
                    # happened (actual reduction much worse than predicted),
                    # the model is no longer locally valid -- shrink how much
                    # of the SA-chosen coefficient we actually apply next
                    # step. When agreement is good, relax back toward full
                    # trust. Motivated by measured data showing this gap
                    # grows ~9x for batches that fail to converge, specifically
                    # in the late steps where convergence should be happening.
                    _gap_thresh = float(getattr(self.args, "ising_trust_gap_threshold", 0.05))
                    _shrink = float(getattr(self.args, "ising_trust_shrink", 0.7))
                    _grow = float(getattr(self.args, "ising_trust_grow", 1.05))
                    _trust_min = float(getattr(self.args, "ising_trust_min", 0.1))
                    if _gap > _gap_thresh:
                        _trust_scale = max(_trust_scale * _shrink, _trust_min)
                    else:
                        _trust_scale = min(_trust_scale * _grow, 1.0)
            residual_history.append(r_prev.detach())
            update_history.append((delta.detach(), (r_next - r_prev).detach()))
            if len(residual_history) > self.args.history_size:
                residual_history = residual_history[-self.args.history_size :]
            if len(update_history) > self.args.history_size:
                update_history = update_history[-self.args.history_size :]
            final_z = z.clone()
            _ = z_prev
        else:
            final_z = z.clone()

        r = residual_at(final_z)
        fz = r + final_z
        final_abs = float(r.norm().item())
        final_rel = float(final_abs / (fz.norm().item() + 1e-9))
        abs_trace.append(final_abs)
        rel_trace.append(final_rel)
        if len(rel_trace) >= threshold:
            idxs = sorted(set([0, len(rel_trace)//4, len(rel_trace)//2, 3*len(rel_trace)//4, len(rel_trace)-1]))
            trace_str = ','.join(str(i) + ':' + format(rel_trace[i], '.5f') for i in idxs)
            print('[ISING-CAPPED-TRACE] tag=' + tag + ' n=' + str(len(rel_trace)) + ' trace=' + trace_str, flush=True)


        t_total = t_residual + t_basis + t_jvp + t_matrix + t_sa + t_candidate
        print(
            f"[ISING-TIMING] tag={tag} nstep={len(rel_trace)} final_rel={final_rel:.6f} total={t_total:.4f}s | "
            f"residual_eval={t_residual:.4f}s({100*t_residual/max(t_total,1e-9):.0f}%) "
            f"jvp_eval={t_jvp:.4f}s({100*t_jvp/max(t_total,1e-9):.0f}%) "
            f"basis={t_basis:.4f}s({100*t_basis/max(t_total,1e-9):.0f}%) "
            f"matrix_build={t_matrix:.4f}s({100*t_matrix/max(t_total,1e-9):.0f}%) "
            f"sa_solve={t_sa:.4f}s({100*t_sa/max(t_total,1e-9):.0f}%) "
            f"candidate_score={t_candidate:.4f}s({100*t_candidate/max(t_total,1e-9):.0f}%)",
            flush=True,
        )

        return {
            "result": final_z,
            "final_abs": final_abs,
            "final_rel": final_rel,
            "nstep": len(rel_trace),
            "abs_trace": abs_trace,
            "rel_trace": rel_trace,
            "t_residual": t_residual,
            "t_basis": t_basis,
            "t_jvp": t_jvp,
            "t_matrix": t_matrix,
            "t_sa": t_sa,
            "t_candidate": t_candidate,
        }


# ---------------------------------------------------------------------------
# W8A8 fake-quantization (ported verbatim from
# NormMatchTrans/model/deq_ising_quant.py) -- architecture agnostic.
# ---------------------------------------------------------------------------

def _sym_fake_quant(x: torch.Tensor, bits: int, per_channel: bool) -> torch.Tensor:
    qmax = float(2 ** (bits - 1) - 1)
    if per_channel:
        scale = x.abs().amax(dim=tuple(range(1, x.dim())), keepdim=True).clamp(min=1e-8) / qmax
    else:
        scale = x.abs().amax().clamp(min=1e-8) / qmax
    x_q = (x / scale).clamp(-qmax, qmax).round() * scale
    return x + (x_q - x).detach()


class QuantizedLinear(nn.Module):
    def __init__(self, linear: nn.Linear, bits: int = 8) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bits = bits
        self.quant_enabled = True
        self.weight = linear.weight
        if linear.bias is not None:
            self.bias = linear.bias
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_enabled:
            x_q = _sym_fake_quant(x, self.bits, per_channel=False)
            w_q = _sym_fake_quant(self.weight, self.bits, per_channel=True)
            return F.linear(x_q, w_q, self.bias)
        return F.linear(x, self.weight, self.bias)


class QuantizedConv1d(nn.Module):
    """iTransformer's FFN uses kernel_size=1 Conv1d instead of Linear; same fake-quant applies."""

    def __init__(self, conv: nn.Conv1d, bits: int = 8) -> None:
        super().__init__()
        assert conv.kernel_size == (1,), "only kernel_size=1 conv (pointwise FFN) is supported"
        self.bits = bits
        self.quant_enabled = True
        self.stride = conv.stride
        self.padding = conv.padding
        self.weight = conv.weight
        self.bias = conv.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_enabled:
            x_q = _sym_fake_quant(x, self.bits, per_channel=False)
            w_q = _sym_fake_quant(self.weight, self.bits, per_channel=True)
            return F.conv1d(x_q, w_q, self.bias, stride=self.stride, padding=self.padding)
        return F.conv1d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)


def replace_linear_recursive(module: nn.Module, bits: int) -> None:
    """Recursively replace nn.Linear and 1x1 nn.Conv1d children with fake-quant versions."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, QuantizedLinear(child, bits))
        elif isinstance(child, nn.Conv1d) and child.kernel_size == (1,):
            setattr(module, name, QuantizedConv1d(child, bits))
        else:
            replace_linear_recursive(child, bits)


def set_quant_enabled(module: nn.Module, enabled: bool) -> None:
    for m in module.modules():
        if isinstance(m, (QuantizedLinear, QuantizedConv1d)):
            m.quant_enabled = enabled


# ---------------------------------------------------------------------------
# iTransformer-specific DEQ wrapper
# ---------------------------------------------------------------------------

class EncoderLayerIsingDEQ(nn.Module):
    """Treats one shared iTransformer EncoderLayer as F_theta and solves
    z* = F_theta(z*, x) with the Ising/SA coefficient solver in the forward
    pass (FP32, quant disabled) and Anderson implicit differentiation in the
    backward pass. W8A8 fake-quant (if enabled via replace_linear_recursive)
    is only switched on for the single "reattach" forward call at z*, mirroring
    NGPTQuantizedIsingDecoder's design.
    """

    def __init__(self, layer: nn.Module, cfg) -> None:
        super().__init__()
        self.layer = layer
        self.norm_inject = nn.LayerNorm(cfg.d_model)
        self.anderson_threshold = int(getattr(cfg, "anderson_threshold", 20))
        self.backward_threshold = int(getattr(cfg, "backward_threshold", 30))
        self.anderson_m = int(getattr(cfg, "anderson_m", 6))
        self.anderson_eps = float(getattr(cfg, "anderson_eps", 1e-3))
        self.anderson_lam = float(getattr(cfg, "anderson_lam", 1e-4))
        self.input_injection_mode = getattr(cfg, "input_injection_mode", "add_norm")
        self.quant_enabled_flag = bool(getattr(cfg, "quant_enabled", False))

        ising_args = SimpleNamespace(
            seed=int(getattr(cfg, "ising_seed", 123)),
            ising_backend=getattr(cfg, "ising_backend", "sa"),
            ising_basis=int(getattr(cfg, "ising_basis", 8)),
            history_size=int(getattr(cfg, "history_size", 4)),
            coeff_bits=int(getattr(cfg, "coeff_bits", 8)),
            ising_coeff_range=float(getattr(cfg, "ising_coeff_range", 2.0)),
            ising_two_stage=bool(getattr(cfg, "ising_two_stage", False)),
            ising_adaptive_range=bool(getattr(cfg, "ising_adaptive_range", False)),
            ising_dynamic_range=bool(getattr(cfg, "ising_dynamic_range", False)),
            ising_dynamic_margin=float(getattr(cfg, "ising_dynamic_margin", 1.5)),
            ising_trust_region=bool(getattr(cfg, "ising_trust_region", False)),
            ising_trust_gap_threshold=float(getattr(cfg, "ising_trust_gap_threshold", 0.05)),
            ising_trust_shrink=float(getattr(cfg, "ising_trust_shrink", 0.7)),
            ising_trust_grow=float(getattr(cfg, "ising_trust_grow", 1.05)),
            ising_trust_min=float(getattr(cfg, "ising_trust_min", 0.1)),
            ising_r_calibration_steps=int(getattr(cfg, "ising_r_calibration_steps", 3)),
            ising_r_margin=float(getattr(cfg, "ising_r_margin", 2.0)),
            ising_r_min=float(getattr(cfg, "ising_r_min", 0.5)),
            ising_r_max=float(getattr(cfg, "ising_r_max", 40.0)),
            ising_probe_diag=bool(getattr(cfg, "ising_probe_diag", False)),
            fd_eps_scale_ratio=(float(cfg.fd_eps_scale_ratio) if getattr(cfg, "fd_eps_scale_ratio", None) is not None else None),
            fd_eps_scale_floor=float(getattr(cfg, "fd_eps_scale_floor", 0.3)),
            ising_central_diff=bool(getattr(cfg, "ising_central_diff", False)),
            ising_secant_probe=bool(getattr(cfg, "ising_secant_probe", False)),
            ising_secant_ratio=float(getattr(cfg, "ising_secant_ratio", 0.5)),
            basis_mode=getattr(cfg, "basis_mode", "mixed_history"),
            fd_eps=float(getattr(cfg, "fd_eps", 1e-3)),
            ising_matrix_scale=float(getattr(cfg, "ising_matrix_scale", 1.0)),
            sa_initial_temperature=float(getattr(cfg, "sa_initial_temperature", 100.0)),
            sa_alpha=float(getattr(cfg, "sa_alpha", 0.95)),
            sa_cutoff_temperature=float(getattr(cfg, "sa_cutoff_temperature", 0.1)),
            sa_iterations_per_t=int(getattr(cfg, "sa_iterations_per_t", 5)),
            sa_restarts=int(getattr(cfg, "sa_restarts", 4)),
            sa_size_limit=int(getattr(cfg, "sa_size_limit", 100)),
            kaiwu_user_id=getattr(cfg, "kaiwu_user_id", None),
            kaiwu_sdk_code=getattr(cfg, "kaiwu_sdk_code", None),
            cim_project_no=getattr(cfg, "cim_project_no", "YOUR_CIM_PROJECT"),
            cim_task_prefix=getattr(cfg, "cim_task_prefix", "qdeq_ts_cim"),
            cim_priority=int(getattr(cfg, "cim_priority", 10)),
            cim_interval=float(getattr(cfg, "cim_interval", 0.01)),
            cim_precision_bits=int(getattr(cfg, "cim_precision_bits", 14)),
            cim_truncated_precision=int(getattr(cfg, "cim_truncated_precision", 18)),
            cim_max_retries=int(getattr(cfg, "cim_max_retries", 20)),
            cim_save_dir=getattr(cfg, "cim_save_dir", "cim_tasks"),
        )
        self.ising_solver = DynamicCoeffIsingSolver(
            ising_args, variant=getattr(cfg, "ising_variant", "residual_scaled")
        )
        self.hook = None
        self.last_solver_info = {}
        self.forward_call_counter = 0

    def inject_input(self, z: torch.Tensor, source_anchor: torch.Tensor) -> torch.Tensor:
        if self.input_injection_mode == "none":
            return z
        if self.input_injection_mode == "add_norm":
            return self.norm_inject(z + source_anchor)
        raise ValueError(f"Unsupported input_injection_mode={self.input_injection_mode!r}")

    def explicit_once(self, z: torch.Tensor, source_anchor: torch.Tensor) -> torch.Tensor:
        layer_input = self.inject_input(z, source_anchor)
        out, _ = self.layer(layer_input, attn_mask=None)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        source_anchor = x

        if self.quant_enabled_flag:
            set_quant_enabled(self.layer, False)

        def f(z):
            return self.explicit_once(z, source_anchor)

        self.forward_call_counter += 1
        mode = "train" if self.training else "eval"
        with torch.no_grad():
            x_norm = float(x.norm().item())
            x_mean = float(x.mean().item())
        tag = f"fwd{self.forward_call_counter}_{mode}_xnorm{x_norm:.3f}_xmean{x_mean:.3f}"
        print(f"[Q-DEQ-Ising-DIAG] === forward_call={self.forward_call_counter} mode={mode} "
              f"batch_shape={tuple(x.shape)} x_norm={x_norm:.4g} x_mean={x_mean:.4g} ===", flush=True)

        # Dropout in self.layer would corrupt the finite-difference JVP: two
        # calls to f() a few epsilon apart would differ mostly by independent
        # dropout masks, not by the true derivative, and dividing by fd_eps
        # (1e-3) amplifies that noise ~1000x. Force eval mode for the solve;
        # the reattach call below still uses the model's real train/eval state.
        was_training = self.layer.training
        self.layer.eval()
        with torch.no_grad():
            result = self.ising_solver.solve(
                f, x, threshold=self.anderson_threshold, eps=self.anderson_eps, stop_mode="rel", tag=tag
            )
        if was_training:
            self.layer.train()

        z_star = result.pop("result")
        info = {"forward_" + key: value for key, value in result.items()}
        info["input_injection_mode"] = self.input_injection_mode

        if self.quant_enabled_flag:
            set_quant_enabled(self.layer, True)

        z_star.requires_grad_()
        new_z = self.explicit_once(z_star, source_anchor)

        if self.training:
            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                    self.hook = None
                grad_f = lambda y: autograd.grad(new_z, z_star, y, retain_graph=True)[0] + grad
                b_result = original_anderson(
                    grad_f,
                    torch.zeros_like(grad),
                    threshold=self.backward_threshold,
                    eps=self.anderson_eps,
                    m=self.anderson_m,
                    lam=self.anderson_lam,
                    stop_mode="rel",
                )
                info["backward_final_rel"] = b_result["final_rel"].detach()
                info["backward_nstep"] = b_result["nstep"]
                return b_result["result"]

            self.hook = new_z.register_hook(backward_hook)

        self.last_solver_info = info
        return new_z


class EncoderLayerAndersonDEQ(nn.Module):
    """Continuous DEQ-Anderson baseline (always FP32, no Ising/SA, no
    quantization) for the same shared EncoderLayer -- mirrors
    NormMatchTrans's NGPTAndersonDecoder: both forward and backward solves
    use ``original_anderson``. Used as the middle comparison point between
    the plain multi-layer iTransformer baseline and the full Q-DEQ-Ising-SA
    variant, isolating how much of any accuracy change comes from the
    shared-layer DEQ architecture itself vs. the Ising-SA + W8A8 addition.
    """

    def __init__(self, layer: nn.Module, cfg) -> None:
        super().__init__()
        self.layer = layer
        self.norm_inject = nn.LayerNorm(cfg.d_model)
        self.anderson_threshold = int(getattr(cfg, "anderson_threshold", 20))
        self.backward_threshold = int(getattr(cfg, "backward_threshold", 30))
        self.anderson_m = int(getattr(cfg, "anderson_m", 6))
        self.anderson_eps = float(getattr(cfg, "anderson_eps", 1e-3))
        self.anderson_lam = float(getattr(cfg, "anderson_lam", 1e-4))
        self.input_injection_mode = getattr(cfg, "input_injection_mode", "add_norm")
        self.jac_loss_weight = float(getattr(cfg, "jac_loss_weight", 0.0))
        self.hook = None
        self.last_solver_info = {}
        self.last_jac_loss = None

    def inject_input(self, z: torch.Tensor, source_anchor: torch.Tensor) -> torch.Tensor:
        if self.input_injection_mode == "none":
            return z
        if self.input_injection_mode == "add_norm":
            return self.norm_inject(z + source_anchor)
        raise ValueError(f"Unsupported input_injection_mode={self.input_injection_mode!r}")

    def explicit_once(self, z: torch.Tensor, source_anchor: torch.Tensor) -> torch.Tensor:
        layer_input = self.inject_input(z, source_anchor)
        out, _ = self.layer(layer_input, attn_mask=None)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        source_anchor = x

        def f(z):
            return self.explicit_once(z, source_anchor)

        with torch.no_grad():
            result = original_anderson(
                f, x,
                threshold=self.anderson_threshold,
                eps=self.anderson_eps,
                m=self.anderson_m,
                lam=self.anderson_lam,
                stop_mode="rel",
            )

        z_star = result.pop("result")
        info = {"forward_" + key: value for key, value in result.items()}
        info["input_injection_mode"] = self.input_injection_mode

        z_star.requires_grad_()
        new_z = self.explicit_once(z_star, source_anchor)

        if self.training and self.jac_loss_weight > 0:
            self.last_jac_loss = jac_loss_estimate(new_z, z_star, vecs=1)
        else:
            self.last_jac_loss = None

        if self.training:
            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                    self.hook = None
                grad_f = lambda y: autograd.grad(new_z, z_star, y, retain_graph=True)[0] + grad
                b_result = original_anderson(
                    grad_f,
                    torch.zeros_like(grad),
                    threshold=self.backward_threshold,
                    eps=self.anderson_eps,
                    m=self.anderson_m,
                    lam=self.anderson_lam,
                    stop_mode="rel",
                )
                info["backward_final_rel"] = b_result["final_rel"].detach()
                info["backward_nstep"] = b_result["nstep"]
                return b_result["result"]

            self.hook = new_z.register_hook(backward_hook)

        self.last_solver_info = info
        return new_z
