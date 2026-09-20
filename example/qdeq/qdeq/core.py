"""The qdeq DEQ fixed-point plugin: QDEQ."""

from __future__ import annotations

import contextlib
from typing import Callable, ContextManager

import torch
import torch.nn as nn

from qdeq.ising_sa import IsingSolverConfig
from qdeq.solvers import Solver, anderson_solver, ising_sa_solver


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


class _DEQImplicitGrad(torch.autograd.Function):
    """Identity in the forward direction; in backward, replaces the incoming
    gradient at ``new_z`` with the solution of the linear fixed-point system
    (I - Jᵀ) y = grad_output, where J = ∂new_z/∂z_star. Autograd then keeps
    flowing this corrected gradient through ``new_z``'s own graph (i.e. through
    func's parameters and any other func_args) exactly as it would for an
    ordinary op — this is the implicit function theorem backward pass from
    Bai et al. 2019, Theorem 1, relocated from a tensor ``register_hook`` (this
    project's existing style) onto a torch.autograd.Function node.
    """

    @staticmethod
    def forward(ctx, new_z, z_star, b_solver, b_thres, stop_mode, info, b_eps):
        ctx.b_solver = b_solver
        ctx.b_thres = b_thres
        ctx.stop_mode = stop_mode
        ctx.info = info
        ctx.b_eps = b_eps
        ctx.save_for_backward(z_star, new_z)
        return new_z

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
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
    (see ``qdeq.solvers.ising_sa_solver``'s docstring) and only safe as a
    forward-only solver. The backward default is independently Anderson.
    """
    return ising_sa_solver(IsingSolverConfig(solver_variant="v5_dynamic"))


class QDEQ(nn.Module):
    """Wraps an arbitrary fθ(z, *args) with forward fixed-point solve.

    ``func`` may be any callable with signature ``func(z, *func_args) -> z_next``
    where ``z_next`` has the same shape as ``z`` — an ``nn.Module``, a bound
    method, or a plain closure.

    ``f_solver`` defaults to the Ising/SA solver (``qdeq.solvers.ising_sa_solver``
    with ``solver_variant="v5_dynamic"``); ``b_solver`` defaults to
    ``anderson_solver``, not to ``f_solver`` — this "Ising forward, Anderson
    backward" combination is what this project's actual production
    quantized decoder (``NGPTQuantizedIsingDecoder``) uses, verified against
    its source directly. Pass ``f_solver=``/``b_solver=`` explicitly to use
    something else. Real-model solver accuracy and wrapper equivalence must
    be checked separately. The original migration test used a degenerate
    squared-norm loss and mismatched output stages; see the corrected
    ``deq_gm/nmt_deq_ising/tests/qdeq_migration_regression.py``.

    W8A8 quantization toggling is orchestrated automatically: if ``func`` has
    a ``set_quant(bool)`` method (or is a bound method whose ``__self__``
    does), QDEQ calls ``set_quant(False)`` before the no-grad forward solve
    (so iteration runs in FP32) and ``set_quant(True)`` before the single
    differentiable reattach call (so only that one call runs quantized). If
    the forward solve raises, ``set_quant(True)`` is still called before the
    exception propagates, so a failed training step never leaves quantization
    stuck disabled for later calls. This is QDEQ's own design choice, informed
    by but not identical to this project's existing Q-DEQ-Ising code: eval
    mode here never toggles quantization at all (it is left exactly as the
    caller set it, so evaluation reflects true INT8 performance without
    QDEQ's involvement) — this project's actual ``NGPTQuantizedIsingDecoder``
    and the iTransformer ``deq_ising_ts.py::EncoderLayerIsingDEQ`` both
    instead toggle unconditionally in both train and eval (verified against
    both sources directly); QDEQ keeps the eval-never-toggles behavior by
    deliberate choice despite this. If ``func``
    has no ``set_quant`` method, none of this applies and QDEQ is fully
    dtype-agnostic, as before. ``solve_ctx``/``reattach_ctx`` remain available
    underneath for any other external state (dropout, etc.) that isn't
    covered by ``set_quant`` — both fire around the same two phases.

    ``func`` must be deterministic across the no-grad solve and the single
    differentiable reattach call other than through ``set_quant``/
    ``solve_ctx``/``reattach_ctx`` — e.g. put any ``nn.Module`` it wraps in
    ``eval()`` mode (or otherwise disable dropout) before calling this
    module. If ``func`` behaves differently between the two phases for any
    other reason, the implicit-function-theorem assumption
    (``new_z ≈ z_star``) breaks and both the forward residual and the
    backward gradients silently degrade.

    Example — a decoder implementing ``set_quant`` gets FP32-iterate /
    INT8-reattach automatically, with the default Ising forward solver.
    ``QDEQ()`` defaults to training mode (like any ``nn.Module``) regardless
    of ``decoder``'s own mode — call ``.train()``/``.eval()`` on ``deq``
    itself to control which path runs::

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
            f_solver: forward fixed-point solver, ``solve(f, z0, threshold, eps=1e-3, stop_mode="rel", **kw) -> dict``.
                Defaults to the Ising/SA solver (``v5_dynamic`` variant) when
                not given.
            b_solver: backward (implicit-gradient) solver; defaults to
                ``anderson_solver`` (NOT to ``f_solver``) — matches this
                project's production ``NGPTQuantizedIsingDecoder``, which
                uses Ising for the forward solve but Anderson for backward.
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
            ``(info["new_z"], info["z_star"])`` to ``qdeq.diagnostics``
            functions for jacobian regularization or a spectral-radius
            estimate; ``z_out`` itself is *not* the right tensor for this —
            its gradient w.r.t. ``z_star`` runs through the implicit-corrected
            backward solve, not the one-step Jacobian these diagnostics
            estimate. Call ``qdeq.diagnostics`` functions *before*
            ``.backward()``, and fold their result into the loss you then
            call ``.backward()`` on — ``new_z``'s graph is freed by that
            call (PyTorch's normal behavior), so using it afterwards raises
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
