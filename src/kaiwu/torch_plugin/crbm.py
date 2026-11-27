# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Conditional Restricted Boltzmann Machine (CRBM).

This module implements a general Conditional Restricted Boltzmann Machine with
binary visible units ``v``, binary hidden units ``h``, and external conditioning
inputs ``c`` (binary or continuous). The conditional energy is

.. math::

    E(v, h \\mid c) = - v^T W h - c^T U h - b_v^T v - b_h^T h - c^T A v

where ``A`` (condition-to-visible direct weights) can be enabled or disabled via
``use_cond_visible``. Folding ``c`` into effective linear biases
``b_v_eff = b_v + A^T c`` and ``b_h_eff = b_h + U^T c`` reduces the conditional
model to an equivalent RBM whose Ising matrix can be sampled directly by Kaiwu
SDK optimizers (simulated annealing or CIM).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as nn_func

from .abstract_boltzmann_machine import AbstractBoltzmannMachine
from .usage_stats import kpp_caller_context

__all__ = ["ConditionalRestrictedBoltzmannMachine"]


def _validate_positive_int(value: int, name: str) -> int:
    """Validate that ``value`` is a positive integer (excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _validate_range(
    bounds: Sequence[float] | None,
    name: str,
) -> tuple[float, float] | None:
    """Validate a ``(low, high)`` finite numeric clipping interval."""
    if bounds is None:
        return None
    if isinstance(bounds, (str, bytes)) or len(bounds) != 2:
        raise ValueError(f"{name} must be a 2-element sequence (low, high)")
    low, high = bounds
    if isinstance(low, bool) or isinstance(high, bool):
        raise ValueError(f"{name} bounds must be numeric, not bool")
    low_f, high_f = float(low), float(high)
    if not (math.isfinite(low_f) and math.isfinite(high_f)):
        raise ValueError(f"{name} bounds must be finite numbers, got {bounds!r}")
    if low_f > high_f:
        raise ValueError(f"{name} lower bound {low_f} exceeds upper bound {high_f}")
    return (low_f, high_f)


def _init_parameter_tensor(
    tensor: torch.Tensor | None,
    expected_shape: tuple[int, ...],
    name: str,
    default_fn,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Parameter:
    """Validate an optional initial parameter tensor or create a default one."""
    if tensor is None:
        return torch.nn.Parameter(default_fn())
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None, got {type(tensor).__name__}")
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
        )
    return torch.nn.Parameter(tensor.detach().clone().to(device=device, dtype=dtype))


class ConditionalRestrictedBoltzmannMachine(AbstractBoltzmannMachine):
    """Conditional Restricted Boltzmann Machine (CRBM).

    Models the conditional Boltzmann distribution over binary visible units
    ``v`` and binary hidden units ``h`` given external conditioning vector ``c``:

    .. math::

        E(v, h \\mid c) = - v^T W h - c^T U h - b_v^T v - b_h^T h - c^T A v

    where ``W`` is ``quadratic_coef``, ``U`` is ``cond_hidden_coef``, ``A`` is
    ``cond_visible_coef`` (present when ``use_cond_visible=True``), and
    ``[b_v, b_h]`` is ``linear_bias``.

    Args:
        num_visible (int): Number of binary visible units.
        num_condition (int): Number of conditioning input features.
        num_hidden (int): Number of binary hidden units.
        use_cond_visible (bool, optional): Whether to include direct
            condition-to-visible weights ``A``. Defaults to ``True``.
        quadratic_coef (torch.Tensor, optional): Visible-hidden weight matrix
            ``W`` of shape ``(num_visible, num_hidden)``.
        linear_bias (torch.Tensor, optional): Concatenated visible and hidden
            linear bias ``[b_v, b_h]`` of shape ``(num_visible + num_hidden,)``.
        cond_hidden_coef (torch.Tensor, optional): Condition-to-hidden weight
            matrix ``U`` of shape ``(num_condition, num_hidden)``.
        cond_visible_coef (torch.Tensor, optional): Condition-to-visible weight
            matrix ``A`` of shape ``(num_condition, num_visible)``. Must be
            ``None`` when ``use_cond_visible=False``.
        h_range (tuple[float, float], optional): Default clipping range
            ``(low, high)`` for ``linear_bias``. Applied on initialization when
            provided.
        j_range (tuple[float, float], optional): Default clipping range
            ``(low, high)`` for coupling weights (``quadratic_coef``,
            ``cond_hidden_coef``, and ``cond_visible_coef``). Applied on
            initialization when provided.
        dtype (torch.dtype, optional): Floating-point dtype for model
            parameters. Defaults to ``torch.float32``.
        device (torch.device or str, optional): Device for tensor construction.
            Defaults to CUDA if available, otherwise CPU.
    """

    def __init__(
        self,
        num_visible: int,
        num_condition: int,
        num_hidden: int,
        use_cond_visible: bool = True,
        quadratic_coef: torch.Tensor | None = None,
        linear_bias: torch.Tensor | None = None,
        cond_hidden_coef: torch.Tensor | None = None,
        cond_visible_coef: torch.Tensor | None = None,
        h_range: Sequence[float] | None = None,
        j_range: Sequence[float] | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(device=device)
        self.device = torch.device(self.device)
        self.num_visible = _validate_positive_int(num_visible, "num_visible")
        self.num_condition = _validate_positive_int(num_condition, "num_condition")
        self.num_hidden = _validate_positive_int(num_hidden, "num_hidden")
        self.num_nodes = self.num_visible + self.num_hidden

        if not isinstance(use_cond_visible, bool):
            raise TypeError(
                f"use_cond_visible must be a bool, got {type(use_cond_visible).__name__}"
            )
        if not use_cond_visible and cond_visible_coef is not None:
            raise ValueError(
                "cond_visible_coef cannot be provided when use_cond_visible is False"
            )
        self.use_cond_visible = use_cond_visible

        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError(f"dtype must be a floating-point torch.dtype, got {dtype!r}")

        self.h_range = _validate_range(h_range, "h_range")
        self.j_range = _validate_range(j_range, "j_range")

        self.quadratic_coef = _init_parameter_tensor(
            quadratic_coef,
            (self.num_visible, self.num_hidden),
            "quadratic_coef",
            lambda: torch.randn(
                (self.num_visible, self.num_hidden), device=self.device, dtype=dtype
            )
            * 0.01,
            self.device,
            dtype,
        )
        self.linear_bias = _init_parameter_tensor(
            linear_bias,
            (self.num_nodes,),
            "linear_bias",
            lambda: torch.zeros(self.num_nodes, device=self.device, dtype=dtype),
            self.device,
            dtype,
        )
        self.cond_hidden_coef = _init_parameter_tensor(
            cond_hidden_coef,
            (self.num_condition, self.num_hidden),
            "cond_hidden_coef",
            lambda: torch.randn(
                (self.num_condition, self.num_hidden), device=self.device, dtype=dtype
            )
            * 0.01,
            self.device,
            dtype,
        )
        if self.use_cond_visible:
            self.cond_visible_coef = _init_parameter_tensor(
                cond_visible_coef,
                (self.num_condition, self.num_visible),
                "cond_visible_coef",
                lambda: torch.randn(
                    (self.num_condition, self.num_visible),
                    device=self.device,
                    dtype=dtype,
                )
                * 0.01,
                self.device,
                dtype,
            )
        else:
            self.register_parameter("cond_visible_coef", None)

        if self.h_range is not None or self.j_range is not None:
            self.clip_parameters()

    @property
    def dtype(self) -> torch.dtype:
        """Return the current floating-point dtype of the model parameters."""
        return self.quadratic_coef.dtype

    @property
    def visible_bias(self) -> torch.Tensor:
        """Return the visible bias vector ``b_v``."""
        return self.linear_bias[: self.num_visible]

    @property
    def hidden_bias(self) -> torch.Tensor:
        """Return the hidden bias vector ``b_h``."""
        return self.linear_bias[self.num_visible :]

    def clip_parameters(
        self,
        h_range: Sequence[float] | None = None,
        j_range: Sequence[float] | None = None,
        c_range: Sequence[float] | None = None,
    ) -> None:
        """Clip linear bias and coupling parameters in-place.

        Args:
            h_range (tuple[float, float], optional): Clipping range for
                ``linear_bias``. Falls back to ``self.h_range`` if omitted.
            j_range (tuple[float, float], optional): Clipping range for
                ``quadratic_coef`` (and conditional weights when ``c_range`` is
                omitted). Falls back to ``self.j_range`` if omitted.
            c_range (tuple[float, float], optional): Optional separate clipping
                range for ``cond_hidden_coef`` and ``cond_visible_coef``.
        """
        eff_h = _validate_range(h_range, "h_range") if h_range is not None else self.h_range
        eff_j = _validate_range(j_range, "j_range") if j_range is not None else self.j_range
        eff_c = _validate_range(c_range, "c_range") if c_range is not None else eff_j

        with torch.no_grad():
            if eff_h is not None:
                self.linear_bias.clamp_(*eff_h)
            if eff_j is not None:
                self.quadratic_coef.clamp_(*eff_j)
            if eff_c is not None:
                self.cond_hidden_coef.clamp_(*eff_c)
                if self.cond_visible_coef is not None:
                    self.cond_visible_coef.clamp_(*eff_c)

    def _prepare_tensor(
        self,
        tensor: torch.Tensor,
        expected_dim: int,
        name: str,
    ) -> torch.Tensor:
        """Validate a 2D state tensor and align it to parameter device/dtype."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.ndim != 2 or tensor.shape[1] != expected_dim:
            raise ValueError(
                f"{name} must have shape (B, {expected_dim}), got {tuple(tensor.shape)}"
            )
        return tensor.to(device=self.quadratic_coef.device, dtype=self.quadratic_coef.dtype)

    def _prepare_condition(
        self,
        condition: torch.Tensor | None,
        batch_size: int | None = None,
    ) -> torch.Tensor | None:
        """Validate a 1D or 2D condition tensor and match an optional batch size."""
        if condition is None:
            return None
        if not isinstance(condition, torch.Tensor):
            raise TypeError(
                f"condition must be a torch.Tensor or None, got {type(condition).__name__}"
            )
        if condition.ndim not in (1, 2) or condition.shape[-1] != self.num_condition:
            raise ValueError(
                f"condition must have shape ({self.num_condition},) or "
                f"(B, {self.num_condition}), got {tuple(condition.shape)}"
            )
        cond = condition.to(device=self.quadratic_coef.device, dtype=self.quadratic_coef.dtype)
        if batch_size is None:
            return cond
        if cond.ndim == 1:
            return cond.unsqueeze(0).expand(batch_size, -1)
        if cond.shape[0] == 1 and batch_size != 1:
            return cond.expand(batch_size, -1)
        if cond.shape[0] != batch_size:
            raise ValueError(
                f"condition batch size {cond.shape[0]} does not match state batch size "
                f"{batch_size}"
            )
        return cond

    def effective_biases(
        self,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute condition-folded effective visible and hidden biases.

        For condition ``c``, returns:
        ``b_v_eff = b_v + c @ A`` (or ``b_v`` when ``use_cond_visible=False``) and
        ``b_h_eff = b_h + c @ U``.

        Args:
            condition (torch.Tensor, optional): Condition vector of shape
                ``(num_condition,)`` or ``(B, num_condition)``. When ``None``,
                returns the unconditional biases ``(b_v, b_h)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Effective visible and hidden
            biases with leading batch dimension matching ``condition``.
        """
        if condition is None:
            return self.visible_bias, self.hidden_bias
        cond = self._prepare_condition(condition)
        b_h_eff = self.hidden_bias + cond @ self.cond_hidden_coef
        if self.cond_visible_coef is not None:
            b_v_eff = self.visible_bias + cond @ self.cond_visible_coef
        elif cond.ndim == 1:
            b_v_eff = self.visible_bias
        else:
            b_v_eff = self.visible_bias.unsqueeze(0).expand(cond.shape[0], -1)
        return b_v_eff, b_h_eff

    def energy(
        self,
        s_visible: torch.Tensor,
        s_hidden: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the conditional Hamiltonian ``E(v, h | c)``.

        Args:
            s_visible (torch.Tensor): Visible states of shape ``(B, num_visible)``.
            s_hidden (torch.Tensor): Hidden states of shape ``(B, num_hidden)``.
            condition (torch.Tensor, optional): Conditioning inputs of shape
                ``(num_condition,)`` or ``(B, num_condition)``.

        Returns:
            torch.Tensor: Conditional energy of shape ``(B,)``.
        """
        vis = self._prepare_tensor(s_visible, self.num_visible, "s_visible")
        hid = self._prepare_tensor(s_hidden, self.num_hidden, "s_hidden")
        if vis.shape[0] != hid.shape[0]:
            raise ValueError(
                f"s_visible batch size {vis.shape[0]} does not match s_hidden "
                f"batch size {hid.shape[0]}"
            )
        cond = self._prepare_condition(condition, batch_size=vis.shape[0])
        b_v_eff, b_h_eff = self.effective_biases(cond)
        interaction = torch.sum((vis @ self.quadratic_coef) * hid, dim=-1)
        vis_term = torch.sum(vis * b_v_eff, dim=-1)
        hid_term = torch.sum(hid * b_h_eff, dim=-1)
        return -interaction - vis_term - hid_term

    def forward(
        self,
        s_all: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the conditional Hamiltonian on concatenated states ``[v, h]``.

        Args:
            s_all (torch.Tensor): Joint state tensor of shape
                ``(B, num_visible + num_hidden)``.
            condition (torch.Tensor, optional): Conditioning inputs of shape
                ``(num_condition,)`` or ``(B, num_condition)``.

        Returns:
            torch.Tensor: Conditional Hamiltonian of shape ``(B,)``.
        """
        states = self._prepare_tensor(s_all, self.num_nodes, "s_all")
        return self.energy(
            states[:, : self.num_visible],
            states[:, self.num_visible :],
            condition=condition,
        )

    def free_energy(
        self,
        s_visible: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the exact conditional free energy ``F(v | c)``.

        Marginalizes analytically over all binary hidden configurations:

        .. math::

            F(v \\mid c) = - b_v^{\\text{eff}}(c)^T v
                          - \\sum_{j=1}^{N_h} \\operatorname{softplus}(
                              b_{h,j}^{\\text{eff}}(c) + (v W)_j)

        Args:
            s_visible (torch.Tensor): Visible states of shape ``(B, num_visible)``.
            condition (torch.Tensor, optional): Conditioning inputs of shape
                ``(num_condition,)`` or ``(B, num_condition)``.

        Returns:
            torch.Tensor: Conditional free energy of shape ``(B,)``.
        """
        vis = self._prepare_tensor(s_visible, self.num_visible, "s_visible")
        cond = self._prepare_condition(condition, batch_size=vis.shape[0])
        b_v_eff, b_h_eff = self.effective_biases(cond)
        vis_term = -torch.sum(vis * b_v_eff, dim=-1)
        hid_pre = vis @ self.quadratic_coef + b_h_eff
        hid_term = -torch.sum(nn_func.softplus(hid_pre), dim=-1)
        return vis_term + hid_term

    def get_hidden(
        self,
        s_visible: torch.Tensor,
        condition: torch.Tensor | None = None,
        requires_grad: bool = False,
        bernoulli: bool = False,
    ) -> torch.Tensor:
        """Propagate visible units and condition to hidden activations ``P(h | v, c)``.

        Args:
            s_visible (torch.Tensor): Visible units of shape ``(B, num_visible)``.
            condition (torch.Tensor, optional): Conditioning inputs of shape
                ``(num_condition,)`` or ``(B, num_condition)``.
            requires_grad (bool, optional): Whether to track gradients.
                Defaults to ``False``.
            bernoulli (bool, optional): If ``True``, sample binary hidden states
                from the Bernoulli probabilities; otherwise return probabilities.
                Defaults to ``False``.

        Returns:
            torch.Tensor: Concatenated states ``[v, h]`` of shape
            ``(B, num_visible + num_hidden)``.
        """
        with torch.set_grad_enabled(requires_grad):
            vis = self._prepare_tensor(s_visible, self.num_visible, "s_visible")
            cond = self._prepare_condition(condition, batch_size=vis.shape[0])
            _, b_h_eff = self.effective_biases(cond)
            prob = torch.sigmoid(vis @ self.quadratic_coef + b_h_eff)
            hid = (prob > torch.rand_like(prob)).to(dtype=prob.dtype) if bernoulli else prob
            return torch.cat([vis, hid], dim=-1)

    def get_visible(
        self,
        s_hidden: torch.Tensor,
        condition: torch.Tensor | None = None,
        requires_grad: bool = False,
        bernoulli: bool = False,
    ) -> torch.Tensor:
        """Propagate hidden units and condition to visible activations ``P(v | h, c)``.

        Args:
            s_hidden (torch.Tensor): Hidden units of shape ``(B, num_hidden)``.
            condition (torch.Tensor, optional): Conditioning inputs of shape
                ``(num_condition,)`` or ``(B, num_condition)``.
            requires_grad (bool, optional): Whether to track gradients.
                Defaults to ``False``.
            bernoulli (bool, optional): If ``True``, sample binary visible states
                from the Bernoulli probabilities; otherwise return probabilities.
                Defaults to ``False``.

        Returns:
            torch.Tensor: Concatenated states ``[v, h]`` of shape
            ``(B, num_visible + num_hidden)``.
        """
        with torch.set_grad_enabled(requires_grad):
            hid = self._prepare_tensor(s_hidden, self.num_hidden, "s_hidden")
            cond = self._prepare_condition(condition, batch_size=hid.shape[0])
            b_v_eff, _ = self.effective_biases(cond)
            prob = torch.sigmoid(hid @ self.quadratic_coef.t() + b_v_eff)
            vis = (prob > torch.rand_like(prob)).to(dtype=prob.dtype) if bernoulli else prob
            return torch.cat([vis, hid], dim=-1)

    def objective(
        self,
        s_positive: torch.Tensor,
        s_negative: torch.Tensor,
        condition: torch.Tensor | None = None,
        condition_negative: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the contrastive conditional NLL gradient objective.

        Returns ``E[E(s_positive | c)] - E[E(s_negative | c_neg)]``, whose
        gradient with respect to the model parameters matches the gradient of
        the conditional negative log-likelihood when ``s_negative`` is drawn
        from ``p(v, h | c_neg)``.

        Args:
            s_positive (torch.Tensor): Positive-phase joint states ``[v, h]`` of
                shape ``(B1, num_nodes)``.
            s_negative (torch.Tensor): Negative-phase joint states ``[v, h]`` of
                shape ``(B2, num_nodes)``.
            condition (torch.Tensor, optional): Conditions for ``s_positive``
                (and also for ``s_negative`` when ``condition_negative`` is
                ``None``).
            condition_negative (torch.Tensor, optional): Conditions paired with
                ``s_negative``. Use when negative sampling produces a different
                batch size than ``s_positive`` (e.g., multiple reads per
                condition from :meth:`condition_sample`).

        Returns:
            torch.Tensor: Scalar objective value.
        """
        neg_cond = condition if condition_negative is None else condition_negative
        return self(s_positive, condition=condition).mean() - self(
            s_negative, condition=neg_cond
        ).mean()

    def _resolve_gibbs_init(
        self,
        condition: torch.Tensor | None,
        s_visible: torch.Tensor | None,
        num_samples: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Resolve initial visible states and aligned conditions for Gibbs sampling."""
        if s_visible is not None:
            vis = self._prepare_tensor(s_visible, self.num_visible, "s_visible")
            batch_size = vis.shape[0]
            if num_samples is not None and num_samples != batch_size:
                raise ValueError(
                    f"num_samples ({num_samples}) must match s_visible batch size ({batch_size})"
                )
            return vis, self._prepare_condition(condition, batch_size=batch_size)

        raw_cond = self._prepare_condition(condition)
        if raw_cond is None:
            if num_samples is None:
                raise ValueError(
                    "At least one of s_visible, condition, or num_samples must be provided"
                )
            batch_size = num_samples
            cond = None
        elif raw_cond.ndim == 1:
            batch_size = 1 if num_samples is None else num_samples
            cond = raw_cond.unsqueeze(0).expand(batch_size, -1)
        elif num_samples is None:
            batch_size = raw_cond.shape[0]
            cond = raw_cond
        elif raw_cond.shape[0] in (1, num_samples):
            batch_size = num_samples
            cond = raw_cond.expand(batch_size, -1)
        else:
            raise ValueError(
                f"num_samples ({num_samples}) conflicts with condition batch size "
                f"({raw_cond.shape[0]})"
            )

        vis = torch.bernoulli(
            torch.full(
                (batch_size, self.num_visible),
                0.5,
                device=self.quadratic_coef.device,
                dtype=self.quadratic_coef.dtype,
            )
        )
        return vis, cond

    def gibbs_sample(
        self,
        condition: torch.Tensor | None = None,
        num_steps: int = 1,
        s_visible: torch.Tensor | None = None,
        num_samples: int | None = None,
        return_prob: bool = False,
    ) -> torch.Tensor:
        """Run blocked alternating Gibbs sampling conditioned on ``condition``.

        Args:
            condition (torch.Tensor, optional): Conditioning inputs of shape
                ``(num_condition,)`` or ``(B, num_condition)``.
            num_steps (int, optional): Number of full ``v -> h -> v`` Gibbs
                steps. Defaults to ``1``.
            s_visible (torch.Tensor, optional): Initial visible states of shape
                ``(B, num_visible)``. If ``None``, initialized from
                ``Bernoulli(0.5)``.
            num_samples (int, optional): Number of samples when ``s_visible`` is
                ``None``. Inferred from ``condition`` batch size when omitted.
            return_prob (bool, optional): If ``True``, the final visible and
                hidden update returns activation probabilities instead of binary
                samples. Defaults to ``False``.

        Returns:
            torch.Tensor: Sampled joint states ``[v, h]`` of shape
            ``(B, num_visible + num_hidden)``.
        """
        steps = _validate_positive_int(num_steps, "num_steps")
        if num_samples is not None:
            _validate_positive_int(num_samples, "num_samples")

        with torch.no_grad():
            vis, cond = self._resolve_gibbs_init(condition, s_visible, num_samples)
            b_v_eff, b_h_eff = self.effective_biases(cond)
            for step in range(steps):
                h_prob = torch.sigmoid(vis @ self.quadratic_coef + b_h_eff)
                hid = (h_prob > torch.rand_like(h_prob)).to(dtype=h_prob.dtype)
                v_prob = torch.sigmoid(hid @ self.quadratic_coef.t() + b_v_eff)
                if return_prob and step == steps - 1:
                    vis = v_prob
                else:
                    vis = (v_prob > torch.rand_like(v_prob)).to(dtype=v_prob.dtype)

            final_h_prob = torch.sigmoid(vis @ self.quadratic_coef + b_h_eff)
            final_hid = (
                final_h_prob
                if return_prob
                else (final_h_prob > torch.rand_like(final_h_prob)).to(
                    dtype=final_h_prob.dtype
                )
            )
            return torch.cat([vis, final_hid], dim=-1)

    def _build_ising_matrix(self, effective_linear_bias: torch.Tensor) -> np.ndarray:
        """Construct the symmetric ``(N+1, N+1)`` Ising matrix for a bias vector."""
        param_device = self.quadratic_coef.device
        param_dtype = self.quadratic_coef.dtype
        work_dtype = torch.float32 if param_dtype == torch.bfloat16 else param_dtype

        with torch.no_grad():
            quad = self.quadratic_coef.to(dtype=work_dtype)
            bias = effective_linear_bias.to(device=param_device, dtype=work_dtype)
            ising_mat = torch.zeros(
                (self.num_nodes + 1, self.num_nodes + 1),
                device=param_device,
                dtype=work_dtype,
            )
            ising_mat[: self.num_visible, self.num_visible : -1] = quad / 8.0
            ising_mat[self.num_visible : -1, : self.num_visible] = quad.t() / 8.0
            ising_bias = bias / 4.0 + ising_mat.sum(dim=0)[:-1]
            ising_mat[: self.num_nodes, -1] = ising_bias
            ising_mat[-1, : self.num_nodes] = ising_bias
            return ising_mat.detach().cpu().numpy()

    def _to_ising_matrix(self) -> np.ndarray:
        """Convert the unconditional (``c = 0``) CRBM to an Ising matrix."""
        return self._build_ising_matrix(self.linear_bias)

    def condition_ising_matrix(
        self,
        condition: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Build the Ising matrix with condition ``c`` folded into linear biases.

        For a single condition vector of shape ``(num_condition,)`` (or ``None``),
        returns a 2D array of shape ``(num_nodes + 1, num_nodes + 1)``. For a
        batch of conditions of shape ``(B, num_condition)``, returns a 3D array
        of shape ``(B, num_nodes + 1, num_nodes + 1)``.

        For any binary state ``s = [v, h]`` and auxiliary spin
        ``x_aux in {-1, +1}`` with ``x = [x_aux * (2 * s - 1), x_aux]``, the
        Ising energy ``-x^T J x`` equals ``E(v, h | c)`` up to a state-independent
        constant.

        Args:
            condition (torch.Tensor, optional): Conditioning vector of shape
                ``(num_condition,)`` or batch of shape ``(B, num_condition)``.

        Returns:
            np.ndarray: Ising matrix or batch of Ising matrices.
        """
        if condition is None:
            return self._to_ising_matrix()
        cond = self._prepare_condition(condition)
        b_v_eff, b_h_eff = self.effective_biases(cond)
        if cond.ndim == 1:
            eff_bias = torch.cat([b_v_eff, b_h_eff], dim=-1)
            return self._build_ising_matrix(eff_bias)
        if cond.shape[0] == 0:
            raise ValueError("condition batch must be non-empty")
        matrices = [
            self._build_ising_matrix(torch.cat([b_v_eff[i], b_h_eff[i]], dim=-1))
            for i in range(cond.shape[0])
        ]
        return np.stack(matrices, axis=0)

    def get_ising_matrix(
        self,
        condition: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Return the Ising matrix, optionally conditioned on ``condition``.

        Args:
            condition (torch.Tensor, optional): Conditioning input passed to
                :meth:`condition_ising_matrix`. When ``None``, returns the
                unconditional Ising matrix.

        Returns:
            np.ndarray: Ising matrix of shape ``(num_nodes + 1, num_nodes + 1)``
            or ``(B, num_nodes + 1, num_nodes + 1)``.
        """
        if condition is None:
            return self._to_ising_matrix()
        return self.condition_ising_matrix(condition)

    def _solve_ising(self, sampler, ising_mat: np.ndarray) -> torch.Tensor:
        """Invoke a Kaiwu-compatible sampler and decode binary ``[v, h]`` states."""
        with kpp_caller_context():
            solution = sampler.solve(ising_mat)
        spins = np.atleast_2d(np.asarray(solution, dtype=np.float64))
        if spins.ndim != 2 or spins.shape[0] == 0 or spins.shape[1] != self.num_nodes + 1:
            raise RuntimeError(
                f"sampler must return spin rows of length {self.num_nodes + 1}, "
                f"got shape {spins.shape}"
            )
        binary = (spins[:, :-1] * spins[:, -1:] + 1.0) / 2.0
        return torch.tensor(
            binary,
            dtype=self.quadratic_coef.dtype,
            device=self.quadratic_coef.device,
        )

    def condition_sample(
        self,
        sampler,
        condition: torch.Tensor,
        return_conditions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Sample joint states ``[v, h]`` from ``p(v, h | c)`` via an Ising sampler.

        Supports both a single condition vector of shape ``(num_condition,)`` and
        a batch of conditions of shape ``(B, num_condition)``.

        Args:
            sampler: Kaiwu SDK optimizer (e.g., ``SimulatedAnnealingOptimizer``
                or a CIM optimizer) exposing ``solve(ising_mat)``.
            condition (torch.Tensor): Conditioning vector of shape
                ``(num_condition,)`` or batch of shape ``(B, num_condition)``.
            return_conditions (bool, optional): If ``True``, also return a
                condition tensor aligned row-for-row with the returned samples.
                Defaults to ``False``.

        Returns:
            torch.Tensor or tuple[torch.Tensor, torch.Tensor]: Sampled binary
            states of shape ``(total_reads, num_nodes)``, and optionally the
            matched condition tensor of shape ``(total_reads, num_condition)``.
        """
        cond = self._prepare_condition(condition)
        if cond.ndim == 1:
            ising_mat = self.condition_ising_matrix(cond)
            samples = self._solve_ising(sampler, ising_mat)
            if return_conditions:
                matched = cond.unsqueeze(0).expand(samples.shape[0], -1).clone()
                return samples, matched
            return samples

        if cond.shape[0] == 0:
            raise ValueError("condition batch must be non-empty")
        sample_chunks: list[torch.Tensor] = []
        cond_chunks: list[torch.Tensor] = []
        for idx in range(cond.shape[0]):
            ising_mat = self.condition_ising_matrix(cond[idx])
            chunk = self._solve_ising(sampler, ising_mat)
            sample_chunks.append(chunk)
            if return_conditions:
                cond_chunks.append(cond[idx].unsqueeze(0).expand(chunk.shape[0], -1).clone())
        all_samples = torch.cat(sample_chunks, dim=0)
        if return_conditions:
            return all_samples, torch.cat(cond_chunks, dim=0)
        return all_samples

    def sample(
        self,
        sampler,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample joint states ``[v, h]`` from the CRBM using an Ising sampler.

        When ``condition`` is ``None``, samples from the unconditional (``c = 0``)
        Ising model. When ``condition`` is provided, delegates to
        :meth:`condition_sample`.

        Args:
            sampler: Kaiwu SDK optimizer exposing ``solve(ising_mat)``.
            condition (torch.Tensor, optional): Conditioning vector of shape
                ``(num_condition,)`` or batch of shape ``(B, num_condition)``.

        Returns:
            torch.Tensor: Sampled binary states of shape
            ``(total_reads, num_nodes)``.
        """
        if condition is not None:
            return self.condition_sample(sampler, condition, return_conditions=False)
        return self._solve_ising(sampler, self._to_ising_matrix())
