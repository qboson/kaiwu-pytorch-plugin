"""Unified fixed-point solver interface for the qdeq plugin.

Every solver in this module (and every solver a caller plugs into
``qdeq.core.QDEQ``) shares the call signature::

    solve(f, z0, threshold, eps=1e-3, stop_mode="rel", **kwargs) -> dict

and returns a dict containing at least ``"result"`` (the solution tensor)
and ``"nstep"`` (an int). This lets any two solvers be swapped for
``f_solver``/``b_solver`` without changing caller code.

``QDEQ`` omits the ``threshold`` kwarg entirely when its
``f_thres``/``b_thres`` is ``None`` (its default), so any solver plugged in
here must supply its own usable default for ``threshold`` — it cannot rely
on the caller always providing one, despite the required-looking signature
below (``anderson_solver``'s default is ``50``; ``ising_sa_solver``'s
falls back to its ``IsingSolverConfig.threshold``).
"""

from __future__ import annotations

from typing import Callable, Protocol

import torch


class Solver(Protocol):
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

    Adapted from this project's existing ``deq_gm/lib/solvers.py::anderson``
    (and its ``deq_nlp`` copy) so it can be plugged into ``QDEQ`` as an
    ``f_solver``/``b_solver``. ``z0`` has a batch axis followed by arbitrary feature axes.
    """
    if z0.ndim < 2 or z0.numel() == 0:
        raise ValueError("z0 must have nonempty batch and feature dimensions")
    if threshold < 3 or m < 2:
        raise ValueError("Anderson requires threshold >= 3 and m >= 2")
    if stop_mode not in {"rel", "abs"}:
        raise ValueError("stop_mode must be 'rel' or 'abs'")
    bsz, flat_dim = z0.shape[0], z0[0].numel()
    alternative_mode = "rel" if stop_mode == "abs" else "abs"

    X = torch.zeros(bsz, m, flat_dim, dtype=z0.dtype, device=z0.device)
    F = torch.zeros(bsz, m, flat_dim, dtype=z0.dtype, device=z0.device)

    X[:, 0], F[:, 0] = z0.reshape(bsz, -1), f(z0).reshape(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape_as(z0)).reshape(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=z0.dtype, device=z0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=z0.dtype, device=z0.device)
    y[:, 0] = 1

    trace_dict = {"abs": [], "rel": []}
    lowest_dict = {"abs": 1e8, "rel": 1e8}
    lowest_step_dict = {"abs": 0, "rel": 0}
    lowest_xest = z0.clone()

    for k in range(2, threshold):
        n = min(k, m)

        G = F[:, :n] - X[:, :n]
        H[:, 1 : n + 1, 1 : n + 1] = torch.bmm(
            G, G.transpose(1, 2)
        ) + lam * torch.eye(n, dtype=z0.dtype, device=z0.device)[None]
        alpha = torch.linalg.solve(H[:, : n + 1, : n + 1], y[:, : n + 1])[:, 1 : n + 1, 0]

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (
            alpha[:, None] @ X[:, :n]
        )[:, 0]
        F[:, k % m] = f(X[:, k % m].reshape_as(z0)).reshape(bsz, -1)

        gx = (F[:, k % m] - X[:, k % m]).view_as(z0)
        abs_diff = gx.norm().item()
        rel_diff = abs_diff / (1e-5 + F[:, k % m].norm().item())
        diff_dict = {"abs": abs_diff, "rel": rel_diff}
        trace_dict["abs"].append(abs_diff)
        trace_dict["rel"].append(rel_diff)

        for mode in ["rel", "abs"]:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode:
                    lowest_xest = X[:, k % m].view_as(z0).clone().detach()
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


import dataclasses

from qdeq.ising_sa import CoefficientIsingFixedPointSolver, IsingSolverConfig


def ising_sa_solver(config: IsingSolverConfig) -> Solver:
    """Wrap CoefficientIsingFixedPointSolver behind the shared Solver signature.

    ``config.threshold``/``config.eps`` are the defaults; passing ``threshold``
    or ``eps`` at call time (as ``QDEQ`` does, via ``f_thres``/``b_thres``)
    overrides them for that call without mutating the original config.

    Warning: when used as ``b_solver`` (the backward/implicit-gradient
    solver), ``config.solver_variant="simple_coeff"`` (the dataclass default)
    is residual-magnitude-blind: ``qubo_to_ising`` folds the linear/residual
    term onto the Ising diagonal, where it cancels as a constant in the SA
    energy delta, so the annealer picks coefficients independent of the
    actual backward residual's size. This measured ~25% relative error in
    gradients vs. an exact/Anderson backward solve on a simple test case.
    Use ``config.solver_variant="v5_dynamic"`` (which self-scales its
    coefficient range to the current residual norm) when this solver serves
    as ``b_solver``; ``simple_coeff`` is fine as ``f_solver`` (measured ~3e-4
    relative error there), since forward iterates only need to track
    progress toward a stopping threshold, not solve a linear system exactly.
    """

    def _solve(
        f,
        z0,
        threshold: int | None = None,
        eps: float | None = None,
        stop_mode: str = "rel",
        **kwargs: object,
    ) -> dict:
        cfg = config
        if threshold is not None or eps is not None:
            cfg = dataclasses.replace(
                config,
                threshold=threshold if threshold is not None else config.threshold,
                eps=eps if eps is not None else config.eps,
            )
        return CoefficientIsingFixedPointSolver(cfg).solve(f, z0)

    return _solve
