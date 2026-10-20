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
        f"[ANDERSON-TIMING] tag={tag} nstep={lowest_step[stop_mode]} total={total_time:.4f}s | "
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


class DynamicCoeffIsingSolver:
    """Faithful port of NormMatchTrans's DynamicCoeffIsingSolver (SA backend only)."""

    def __init__(self, args, variant="residual_scaled"):
        self.args = args
        self.variant = variant
        backend = getattr(args, "ising_backend", "sa")
        if backend != "sa":
            raise ValueError(f"only the sa backend is wired up here, got: {backend}")
        self.backend = KaiwuSASolver(args)
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

    def _coeff_spec(self, step, r_norm, m):
        scale = r_norm / math.sqrt(max(1, m))
        bits = self.args.coeff_bits
        if self.variant == "residual_scaled":
            return -2.0, 4.0 / ((2**bits) - 1), bits, scale
        raise ValueError(f"unknown Ising variant: {self.variant}")

    def _basis_coeff_matrix(self, m, coeff_min, coeff_step, bits, scale):
        n = m * bits
        B = np.zeros((m, n), dtype=np.float64)
        offset = np.full(m, coeff_min * scale, dtype=np.float64)
        for i in range(m):
            for j in range(bits):
                B[i, i * bits + j] = scale * coeff_step * (2**j)
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
        update_history = []

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
            fd_eps = self.args.fd_eps
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

            raw_result = np.asarray(result)
            raw_rows = raw_result.shape[0] if raw_result.ndim >= 1 else 1

            if raw_rows > DEBUG_DUMP_ROW_THRESHOLD or solve_elapsed > DEBUG_DUMP_TIME_THRESHOLD_S:
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

            coeff = torch.tensor(coeff_np, dtype=z.dtype, device=z.device)
            delta = sum(coeff[i] * basis[i] for i in range(m))
            z_prev, r_prev = z, r
            z = z + delta
            _t0 = time.perf_counter()
            r_next = residual_at(z)
            t_residual += time.perf_counter() - _t0
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

        t_total = t_residual + t_basis + t_jvp + t_matrix + t_sa + t_candidate
        print(
            f"[ISING-TIMING] tag={tag} nstep={len(rel_trace)} total={t_total:.4f}s | "
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
            basis_mode=getattr(cfg, "basis_mode", "mixed_history"),
            fd_eps=float(getattr(cfg, "fd_eps", 1e-3)),
            ising_matrix_scale=float(getattr(cfg, "ising_matrix_scale", 1.0)),
            sa_initial_temperature=float(getattr(cfg, "sa_initial_temperature", 100.0)),
            sa_alpha=float(getattr(cfg, "sa_alpha", 0.95)),
            sa_cutoff_temperature=float(getattr(cfg, "sa_cutoff_temperature", 0.1)),
            sa_iterations_per_t=int(getattr(cfg, "sa_iterations_per_t", 5)),
            sa_restarts=int(getattr(cfg, "sa_restarts", 4)),
            kaiwu_user_id=getattr(cfg, "kaiwu_user_id", None),
            kaiwu_sdk_code=getattr(cfg, "kaiwu_sdk_code", None),
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

        with torch.no_grad():
            result = self.ising_solver.solve(
                f, x, threshold=self.anderson_threshold, eps=self.anderson_eps, stop_mode="rel", tag=tag
            )

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
        self.hook = None
        self.last_solver_info = {}

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
