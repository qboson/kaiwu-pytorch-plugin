# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Deep Boltzmann Machine (DBM).

This module implements a general multi-layer undirected Deep Boltzmann Machine
with binary visible units ``v`` and ``L >= 1`` binary hidden layers
``h^(1), ..., h^(L)`` coupled bidirectionally between adjacent layers:

.. math::

    E(v, h^{(1)}, \\dots, h^{(L)}) =
        - b_v^T v
        - \\sum_{\\ell=1}^{L} b_{h^{(\\ell)}}^T h^{(\\ell)}
        - v^T W^{(1)} h^{(1)}
        - \\sum_{\\ell=2}^{L} (h^{(\\ell-1)})^T W^{(\\ell)} h^{(\\ell)}

Unlike a greedily stacked Deep Belief Network (DBN), all layers participate in
a single undirected joint Hamiltonian. ``to_ising_matrix`` / ``get_ising_matrix``
maps all visible and hidden layers together into one joint Ising matrix so that
Kaiwu SDK optimizers (simulated annealing or Coherent Ising Machine, CIM) can
jointly sample the entire multi-layer state in one call.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from .abstract_boltzmann_machine import AbstractBoltzmannMachine
from .usage_stats import kpp_caller_context

__all__ = ["DeepBoltzmannMachine"]


def _validate_positive_int(value: int, name: str) -> int:
    """Validate that ``value`` is a positive integer (excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _validate_hidden_dims(hidden_dims: Sequence[int]) -> tuple[int, ...]:
    """Validate a non-empty sequence of positive hidden layer sizes."""
    if isinstance(hidden_dims, (str, bytes)) or not isinstance(hidden_dims, Sequence):
        raise TypeError("hidden_dims must be a non-empty sequence of positive integers")
    if len(hidden_dims) == 0:
        raise ValueError("hidden_dims must contain at least one hidden layer")
    return tuple(
        _validate_positive_int(dim, f"hidden_dims[{idx}]")
        for idx, dim in enumerate(hidden_dims)
    )


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


def _validate_parameter_list(
    tensors: Sequence[torch.Tensor] | None,
    expected_shapes: Sequence[tuple[int, ...]],
    name: str,
    default_factory,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.ParameterList:
    """Validate an optional sequence of parameter tensors or create defaults."""
    if tensors is None:
        return torch.nn.ParameterList(
            [torch.nn.Parameter(default_factory(shape)) for shape in expected_shapes]
        )
    if isinstance(tensors, (str, bytes, torch.Tensor)) or not isinstance(tensors, Sequence):
        raise TypeError(f"{name} must be a sequence of torch.Tensor or None")
    if len(tensors) != len(expected_shapes):
        raise ValueError(
            f"{name} must have length {len(expected_shapes)}, got {len(tensors)}"
        )
    params = []
    for idx, (tensor, shape) in enumerate(zip(tensors, expected_shapes)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{name}[{idx}] must be a torch.Tensor, got {type(tensor).__name__}"
            )
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name}[{idx}] must have shape {shape}, got {tuple(tensor.shape)}"
            )
        params.append(
            torch.nn.Parameter(tensor.detach().clone().to(device=device, dtype=dtype))
        )
    return torch.nn.ParameterList(params)


class DeepBoltzmannMachine(AbstractBoltzmannMachine):
    """Multi-layer undirected Deep Boltzmann Machine (DBM).

    Supports one visible layer and an arbitrary number ``L >= 1`` of hidden
    layers with bidirectional adjacent-layer couplings ``W^(1), ..., W^(L)``
    and no intra-layer edges.

    Args:
        num_visible (int): Number of binary visible units.
        hidden_dims (Sequence[int]): Sizes of hidden layers ``(N_1, ..., N_L)``.
            Must contain at least one positive integer (2+ layers recommended
            for deep multi-layer modeling).
        weights (Sequence[torch.Tensor], optional): Initial coupling matrices
            ``[W^(1), ..., W^(L)]`` with shapes ``(N_{l-1}, N_l)``.
        biases (Sequence[torch.Tensor], optional): Initial bias vectors
            ``[b_v, b_{h^(1)}, ..., b_{h^(L)}]`` with shapes ``(N_l,)``.
        h_range (Sequence[float], optional): Default clipping range ``(low, high)``
            for all layer biases. Applied at initialization when provided.
        j_range (Sequence[float], optional): Default clipping range ``(low, high)``
            for all inter-layer weight matrices. Applied at initialization when
            provided.
        dtype (torch.dtype, optional): Floating-point dtype for parameters.
            Defaults to ``torch.float32``.
        device (torch.device or str, optional): Device for tensor construction.
            Defaults to CUDA if available, otherwise CPU.
    """

    def __init__(
        self,
        num_visible: int,
        hidden_dims: Sequence[int],
        weights: Sequence[torch.Tensor] | None = None,
        biases: Sequence[torch.Tensor] | None = None,
        h_range: Sequence[float] | None = None,
        j_range: Sequence[float] | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(device=device)
        self.device = torch.device(self.device)
        self.num_visible = _validate_positive_int(num_visible, "num_visible")
        self.hidden_dims = _validate_hidden_dims(hidden_dims)
        self.layer_dims = (self.num_visible, *self.hidden_dims)
        self.num_hidden_layers = len(self.hidden_dims)
        self.num_hidden = sum(self.hidden_dims)
        self.num_nodes = self.num_visible + self.num_hidden

        offsets = [0]
        for dim in self.layer_dims:
            offsets.append(offsets[-1] + dim)
        self._layer_offsets = tuple(offsets)

        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError(f"dtype must be a floating-point torch.dtype, got {dtype!r}")

        self.h_range = _validate_range(h_range, "h_range")
        self.j_range = _validate_range(j_range, "j_range")

        weight_shapes = [
            (self.layer_dims[idx], self.layer_dims[idx + 1])
            for idx in range(self.num_hidden_layers)
        ]
        bias_shapes = [(dim,) for dim in self.layer_dims]

        self.weights = _validate_parameter_list(
            weights,
            weight_shapes,
            "weights",
            lambda shape: torch.randn(shape, device=self.device, dtype=dtype) * 0.01,
            self.device,
            dtype,
        )
        self.biases = _validate_parameter_list(
            biases,
            bias_shapes,
            "biases",
            lambda shape: torch.zeros(shape, device=self.device, dtype=dtype),
            self.device,
            dtype,
        )

        if self.h_range is not None or self.j_range is not None:
            self.clip_parameters()

    @property
    def dtype(self) -> torch.dtype:
        """Return the current floating-point dtype of the model parameters."""
        return self.weights[0].dtype

    @property
    def visible_bias(self) -> torch.Tensor:
        """Return the visible layer bias vector ``b_v``."""
        return self.biases[0]

    @property
    def hidden_biases(self) -> tuple[torch.Tensor, ...]:
        """Return a tuple of hidden layer bias vectors ``(b_{h^(1)}, ..., b_{h^(L)})``."""
        return tuple(self.biases[1:])

    def clip_parameters(
        self,
        h_range: Sequence[float] | None = None,
        j_range: Sequence[float] | None = None,
    ) -> None:
        """Clip layer biases and inter-layer coupling weights in-place.

        Args:
            h_range (Sequence[float], optional): Clipping bounds for all layer
                biases. Falls back to ``self.h_range`` when omitted.
            j_range (Sequence[float], optional): Clipping bounds for all
                coupling weights. Falls back to ``self.j_range`` when omitted.
        """
        eff_h = _validate_range(h_range, "h_range") if h_range is not None else self.h_range
        eff_j = _validate_range(j_range, "j_range") if j_range is not None else self.j_range
        with torch.no_grad():
            if eff_h is not None:
                for bias in self.biases:
                    bias.clamp_(*eff_h)
            if eff_j is not None:
                for weight in self.weights:
                    weight.clamp_(*eff_j)

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
        return tensor.to(device=self.weights[0].device, dtype=self.weights[0].dtype)

    def split_states(self, s_all: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Split a concatenated joint tensor ``(B, num_nodes)`` into per-layer tensors.

        Args:
            s_all (torch.Tensor): Joint states ``[v, h^(1), ..., h^(L)]`` of
                shape ``(B, num_nodes)``.

        Returns:
            tuple[torch.Tensor, ...]: Per-layer tensors ``(v, h^(1), ..., h^(L))``.
        """
        states = self._prepare_tensor(s_all, self.num_nodes, "s_all")
        return tuple(
            states[:, self._layer_offsets[idx] : self._layer_offsets[idx + 1]]
            for idx in range(len(self.layer_dims))
        )

    def _normalize_layer_states(
        self,
        states: torch.Tensor | Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Accept either concatenated ``(B, num_nodes)`` or per-layer tensors."""
        if isinstance(states, torch.Tensor):
            return self.split_states(states)
        if isinstance(states, (str, bytes)) or not isinstance(states, Sequence):
            raise TypeError(
                "states must be a torch.Tensor of shape (B, num_nodes) or a sequence "
                "of per-layer tensors"
            )
        if len(states) != len(self.layer_dims):
            raise ValueError(
                f"states must contain {len(self.layer_dims)} layer tensors, got {len(states)}"
            )
        prepared = []
        batch_size = None
        for idx, (layer_tensor, dim) in enumerate(zip(states, self.layer_dims)):
            aligned = self._prepare_tensor(layer_tensor, dim, f"states[{idx}]")
            if batch_size is None:
                batch_size = aligned.shape[0]
            elif aligned.shape[0] != batch_size:
                raise ValueError(
                    f"states[{idx}] batch size {aligned.shape[0]} does not match "
                    f"layer 0 batch size {batch_size}"
                )
            prepared.append(aligned)
        return tuple(prepared)

    def energy(
        self,
        states: torch.Tensor | Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the joint DBM Hamiltonian ``E(v, h^(1), ..., h^(L))``.

        Args:
            states (torch.Tensor or Sequence[torch.Tensor]): Either a
                concatenated state tensor of shape ``(B, num_nodes)`` or a
                sequence of layer tensors ``(v, h^(1), ..., h^(L))``.

        Returns:
            torch.Tensor: Energy values of shape ``(B,)``.
        """
        layers = self._normalize_layer_states(states)
        total = -layers[0] @ self.biases[0]
        for idx in range(self.num_hidden_layers):
            lower = layers[idx]
            upper = layers[idx + 1]
            total = (
                total
                - upper @ self.biases[idx + 1]
                - torch.sum((lower @ self.weights[idx]) * upper, dim=-1)
            )
        return total

    def forward(self, s_all: torch.Tensor) -> torch.Tensor:
        """Compute the joint DBM Hamiltonian on concatenated states ``[v, h^(1), ..., h^(L)]``.

        Args:
            s_all (torch.Tensor): Joint state tensor of shape ``(B, num_nodes)``.

        Returns:
            torch.Tensor: Hamiltonian of shape ``(B,)``.
        """
        return self.energy(s_all)

    def _layer_activation_prob(
        self,
        layer_idx: int,
        layers: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute conditional Bernoulli probabilities for layer ``layer_idx``."""
        pre_act = self.biases[layer_idx].unsqueeze(0).expand(layers[0].shape[0], -1)
        if layer_idx > 0:
            pre_act = pre_act + layers[layer_idx - 1] @ self.weights[layer_idx - 1]
        if layer_idx < self.num_hidden_layers:
            pre_act = pre_act + layers[layer_idx + 1] @ self.weights[layer_idx].t()
        return torch.sigmoid(pre_act)

    def mean_field_inference(
        self,
        s_visible: torch.Tensor,
        num_steps: int = 20,
        tol: float = 1e-6,
        init_means: Sequence[torch.Tensor] | None = None,
        requires_grad: bool = False,
        return_layers: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Run variational mean-field positive-phase inference given visible ``v``.

        Performs sequential block-coordinate updates over hidden layers
        ``l = 1, ..., L``:

        .. math::

            \\mu^{(\\ell)} \\leftarrow \\sigma\\bigl(
                b^{(\\ell)}
                + \\mu^{(\\ell-1)} W^{(\\ell)}
                + \\mathbb{I}[\\ell < L]\\, \\mu^{(\\ell+1)} (W^{(\\ell+1)})^T
            \\bigr)

        Each layer update is an exact coordinate minimization of the variational
        free energy ``F_MF(v, mu) = E_q[E(v, h)] - H(q)``, guaranteeing
        monotonically non-increasing variational free energy at every step.

        Args:
            s_visible (torch.Tensor): Visible states of shape ``(B, num_visible)``.
            num_steps (int, optional): Maximum number of coordinate-descent
                sweeps across hidden layers. Defaults to ``20``.
            tol (float, optional): Early-stopping infinity-norm tolerance on
                mean-field updates. Non-negative; set to ``0.0`` to always run
                all ``num_steps`` sweeps. Defaults to ``1e-6``.
            init_means (Sequence[torch.Tensor], optional): Optional initial
                hidden layer means ``[mu^(1), ..., mu^(L)]``. Defaults to ``0.5``.
            requires_grad (bool, optional): Whether to track gradients through
                the unrolled mean-field updates. Defaults to ``False``.
            return_layers (bool, optional): If ``True``, return a tuple of
                hidden mean tensors ``(mu^(1), ..., mu^(L))``; otherwise return
                concatenated ``[v, mu^(1), ..., mu^(L)]`` of shape
                ``(B, num_nodes)``. Defaults to ``False``.

        Returns:
            torch.Tensor or tuple[torch.Tensor, ...]: Concatenated positive-phase
            mean-field state ``(B, num_nodes)`` or tuple of hidden mean tensors.
        """
        steps = _validate_positive_int(num_steps, "num_steps")
        if (
            isinstance(tol, bool)
            or not isinstance(tol, (int, float))
            or tol < 0
            or not math.isfinite(tol)
        ):
            raise ValueError(f"tol must be a finite non-negative float, got {tol!r}")

        with torch.set_grad_enabled(requires_grad):
            vis = self._prepare_tensor(s_visible, self.num_visible, "s_visible")
            batch_size = vis.shape[0]
            if init_means is None:
                hidden_means = [
                    torch.full(
                        (batch_size, dim),
                        0.5,
                        device=vis.device,
                        dtype=vis.dtype,
                    )
                    for dim in self.hidden_dims
                ]
            else:
                if isinstance(init_means, (str, bytes, torch.Tensor)) or not isinstance(
                    init_means, Sequence
                ):
                    raise TypeError("init_means must be a sequence of hidden mean tensors")
                if len(init_means) != self.num_hidden_layers:
                    raise ValueError(
                        f"init_means must have length {self.num_hidden_layers}, "
                        f"got {len(init_means)}"
                    )
                hidden_means = []
                for idx, (mu_init, dim) in enumerate(zip(init_means, self.hidden_dims)):
                    aligned = self._prepare_tensor(mu_init, dim, f"init_means[{idx}]")
                    if aligned.shape[0] != batch_size:
                        raise ValueError(
                            f"init_means[{idx}] batch size {aligned.shape[0]} does not "
                            f"match s_visible batch size {batch_size}"
                        )
                    hidden_means.append(aligned)

            layers: list[torch.Tensor] = [vis, *hidden_means]
            for _ in range(steps):
                max_delta = 0.0
                for layer_idx in range(1, len(layers)):
                    updated = self._layer_activation_prob(layer_idx, layers)
                    if tol > 0.0:
                        delta = float(
                            (updated - layers[layer_idx]).detach().abs().max().item()
                        )
                        max_delta = max(max_delta, delta)
                    layers[layer_idx] = updated
                if tol > 0.0 and max_delta < tol:
                    break

            if return_layers:
                return tuple(layers[1:])
            return torch.cat(layers, dim=-1)

    def variational_free_energy(
        self,
        s_visible: torch.Tensor,
        hidden_means: Sequence[torch.Tensor] | None = None,
        num_steps: int = 20,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """Compute the mean-field variational free energy ``E_q[E(v, h)] - H(q)``.

        Provides a rigorous upper bound on the exact visible free energy
        ``F(v) = -log sum_h exp(-E(v, h))`` (exact when ``L == 1``).

        Args:
            s_visible (torch.Tensor): Visible states of shape ``(B, num_visible)``.
            hidden_means (Sequence[torch.Tensor], optional): Precomputed hidden
                mean-field probabilities ``(mu^(1), ..., mu^(L))``. If ``None``,
                computed via :meth:`mean_field_inference`.
            num_steps (int, optional): Mean-field iterations when
                ``hidden_means`` is ``None``. Defaults to ``20``.
            tol (float, optional): Mean-field tolerance when ``hidden_means`` is
                ``None``. Defaults to ``1e-6``.

        Returns:
            torch.Tensor: Variational free energy of shape ``(B,)``.
        """
        vis = self._prepare_tensor(s_visible, self.num_visible, "s_visible")
        if hidden_means is None:
            mus = self.mean_field_inference(
                vis, num_steps=num_steps, tol=tol, requires_grad=False, return_layers=True
            )
        else:
            layers = self._normalize_layer_states([vis, *hidden_means])
            mus = layers[1:]

        expected_energy = self.energy([vis, *mus])
        eps = torch.finfo(vis.dtype).eps
        entropy = torch.zeros(vis.shape[0], device=vis.device, dtype=vis.dtype)
        for mu in mus:
            mu_safe = mu.clamp(min=eps, max=1.0 - eps)
            term = -(
                torch.where(mu > 0.0, mu * torch.log(mu_safe), torch.zeros_like(mu))
                + torch.where(
                    mu < 1.0, (1.0 - mu) * torch.log(1.0 - mu_safe), torch.zeros_like(mu)
                )
            )
            entropy = entropy + torch.sum(term, dim=-1)
        return expected_energy - entropy

    def _resolve_gibbs_init(
        self,
        s_all: torch.Tensor | None,
        s_visible: torch.Tensor | None,
        num_samples: int | None,
    ) -> list[torch.Tensor]:
        """Resolve initial per-layer states for alternating block Gibbs sampling."""
        param_device = self.weights[0].device
        param_dtype = self.weights[0].dtype

        if s_all is not None and s_visible is not None:
            raise ValueError("Provide at most one of s_all or s_visible")
        if s_all is not None:
            layers = list(self.split_states(s_all))
            if num_samples is not None and num_samples != layers[0].shape[0]:
                raise ValueError(
                    f"num_samples ({num_samples}) must match s_all batch size "
                    f"({layers[0].shape[0]})"
                )
            return layers
        if s_visible is not None:
            vis = self._prepare_tensor(s_visible, self.num_visible, "s_visible")
            batch_size = vis.shape[0]
            if num_samples is not None and num_samples != batch_size:
                raise ValueError(
                    f"num_samples ({num_samples}) must match s_visible batch size "
                    f"({batch_size})"
                )
            hidden = [
                torch.bernoulli(
                    torch.full((batch_size, dim), 0.5, device=param_device, dtype=param_dtype)
                )
                for dim in self.hidden_dims
            ]
            return [vis, *hidden]
        if num_samples is None:
            raise ValueError("At least one of s_all, s_visible, or num_samples must be provided")
        return [
            torch.bernoulli(
                torch.full((num_samples, dim), 0.5, device=param_device, dtype=param_dtype)
            )
            for dim in self.layer_dims
        ]

    def gibbs_sample(
        self,
        num_steps: int = 1,
        s_all: torch.Tensor | None = None,
        s_visible: torch.Tensor | None = None,
        num_samples: int | None = None,
        clamp_visible: bool = False,
        return_prob: bool = False,
    ) -> torch.Tensor:
        """Run bipartite alternating block Gibbs sampling across odd and even layers.

        Odd layers ``{h^(1), h^(3), ...}`` are conditionally independent given
        even layers ``{v, h^(2), h^(4), ...}`` and vice versa, enabling parallel
        block updates per parity group.

        Args:
            num_steps (int, optional): Number of full odd-then-even Gibbs sweeps.
                Defaults to ``1``.
            s_all (torch.Tensor, optional): Initial full state ``(B, num_nodes)``.
            s_visible (torch.Tensor, optional): Initial visible state
                ``(B, num_visible)`` (hidden layers initialized from
                ``Bernoulli(0.5)``).
            num_samples (int, optional): Number of chains to draw when both
                ``s_all`` and ``s_visible`` are ``None``.
            clamp_visible (bool, optional): If ``True``, keep visible units fixed
                to sample from ``p(h | v)``. Defaults to ``False``.
            return_prob (bool, optional): If ``True``, return activation
                probabilities on the final update of each non-clamped layer.
                Defaults to ``False``.

        Returns:
            torch.Tensor: Sampled joint states ``(B, num_nodes)``.
        """
        steps = _validate_positive_int(num_steps, "num_steps")
        if num_samples is not None:
            _validate_positive_int(num_samples, "num_samples")

        with torch.no_grad():
            layers = self._resolve_gibbs_init(s_all, s_visible, num_samples)
            num_total_layers = len(layers)
            odd_indices = range(1, num_total_layers, 2)
            even_start = 2 if clamp_visible else 0
            even_indices = range(even_start, num_total_layers, 2)

            for step in range(steps):
                is_last = step == steps - 1
                odd_next = {}
                for idx in odd_indices:
                    prob = self._layer_activation_prob(idx, layers)
                    odd_next[idx] = (
                        prob
                        if (return_prob and is_last)
                        else (prob > torch.rand_like(prob)).to(dtype=prob.dtype)
                    )
                for idx, val in odd_next.items():
                    layers[idx] = val

                even_next = {}
                for idx in even_indices:
                    prob = self._layer_activation_prob(idx, layers)
                    even_next[idx] = (
                        prob
                        if (return_prob and is_last)
                        else (prob > torch.rand_like(prob)).to(dtype=prob.dtype)
                    )
                for idx, val in even_next.items():
                    layers[idx] = val

            return torch.cat(layers, dim=-1)

    def reconstruct(
        self,
        s_visible: torch.Tensor,
        num_mf_steps: int = 20,
        tol: float = 1e-6,
        bernoulli: bool = False,
    ) -> torch.Tensor:
        """Reconstruct visible units via multi-layer mean-field inference.

        Runs :meth:`mean_field_inference` so top-down signals from deeper hidden
        layers refine ``mu^(1)``, then computes ``P(v = 1 | mu^(1))``.

        Args:
            s_visible (torch.Tensor): Input visible states ``(B, num_visible)``.
            num_mf_steps (int, optional): Mean-field sweeps. Defaults to ``20``.
            tol (float, optional): Mean-field convergence tolerance.
                Defaults to ``1e-6``.
            bernoulli (bool, optional): If ``True``, sample binary visible units;
                otherwise return reconstruction probabilities. Defaults to ``False``.

        Returns:
            torch.Tensor: Reconstructed visible tensor of shape ``(B, num_visible)``.
        """
        with torch.no_grad():
            hidden_means = self.mean_field_inference(
                s_visible,
                num_steps=num_mf_steps,
                tol=tol,
                requires_grad=False,
                return_layers=True,
            )
            prob = torch.sigmoid(
                self.biases[0].unsqueeze(0) + hidden_means[0] @ self.weights[0].t()
            )
            if bernoulli:
                return (prob > torch.rand_like(prob)).to(dtype=prob.dtype)
            return prob

    def _to_ising_matrix(self) -> np.ndarray:
        """Convert the full multi-layer DBM into one joint Ising matrix.

        Maps all visible and hidden units ``[v, h^(1), ..., h^(L)]`` into a
        single symmetric ``(num_nodes + 1, num_nodes + 1)`` Ising matrix whose
        energy ``-x^T J x`` equals ``E(v, h^(1), ..., h^(L))`` up to a
        state-independent constant for both auxiliary spin signs
        ``x_aux in {-1, +1}``.
        """
        param_device = self.weights[0].device
        param_dtype = self.weights[0].dtype
        work_dtype = torch.float32 if param_dtype == torch.bfloat16 else param_dtype

        with torch.no_grad():
            ising_mat = torch.zeros(
                (self.num_nodes + 1, self.num_nodes + 1),
                device=param_device,
                dtype=work_dtype,
            )
            for idx, weight in enumerate(self.weights):
                lo_start, lo_end = self._layer_offsets[idx], self._layer_offsets[idx + 1]
                hi_start, hi_end = (
                    self._layer_offsets[idx + 1],
                    self._layer_offsets[idx + 2],
                )
                scaled = weight.to(dtype=work_dtype) / 8.0
                ising_mat[lo_start:lo_end, hi_start:hi_end] = scaled
                ising_mat[hi_start:hi_end, lo_start:lo_end] = scaled.t()

            full_bias = torch.cat(
                [bias.to(dtype=work_dtype) for bias in self.biases], dim=0
            )
            ising_bias = full_bias / 4.0 + ising_mat.sum(dim=0)[:-1]
            ising_mat[: self.num_nodes, -1] = ising_bias
            ising_mat[-1, : self.num_nodes] = ising_bias
            return ising_mat.detach().cpu().numpy()

    def to_ising_matrix(self) -> np.ndarray:
        """Return the joint Ising matrix covering all visible and hidden layers.

        Alias for :meth:`get_ising_matrix`.

        Returns:
            np.ndarray: Symmetric Ising matrix of shape
            ``(num_nodes + 1, num_nodes + 1)``.
        """
        return self._to_ising_matrix()

    def clamped_ising_matrix(self, s_visible: torch.Tensor) -> np.ndarray:
        """Build the hidden-only Ising matrix with visible layer clamped to ``v``.

        Folds ``v @ W^(1)`` into the first hidden layer bias
        ``b_{h^(1)}^eff = b_{h^(1)} + v @ W^(1)`` so an Ising sampler (SA or
        CIM) can directly sample the posterior ``p(h^(1), ..., h^(L) | v)``.

        Args:
            s_visible (torch.Tensor): Single visible vector of shape
                ``(num_visible,)`` or ``(1, num_visible)``.

        Returns:
            np.ndarray: Symmetric Ising matrix of shape
            ``(num_hidden + 1, num_hidden + 1)``.
        """
        if not isinstance(s_visible, torch.Tensor):
            raise TypeError(
                f"s_visible must be a torch.Tensor, got {type(s_visible).__name__}"
            )
        if s_visible.ndim == 1 and s_visible.shape[0] == self.num_visible:
            vis = s_visible
        elif s_visible.ndim == 2 and s_visible.shape == (1, self.num_visible):
            vis = s_visible[0]
        else:
            raise ValueError(
                f"s_visible must have shape ({self.num_visible},) or "
                f"(1, {self.num_visible}), got {tuple(s_visible.shape)}"
            )

        param_device = self.weights[0].device
        param_dtype = self.weights[0].dtype
        work_dtype = torch.float32 if param_dtype == torch.bfloat16 else param_dtype

        with torch.no_grad():
            vis_aligned = vis.to(device=param_device, dtype=work_dtype)
            ising_mat = torch.zeros(
                (self.num_hidden + 1, self.num_hidden + 1),
                device=param_device,
                dtype=work_dtype,
            )
            hidden_offsets = [offset - self.num_visible for offset in self._layer_offsets[1:]]
            for idx in range(1, self.num_hidden_layers):
                lo_start, lo_end = hidden_offsets[idx - 1], hidden_offsets[idx]
                hi_start, hi_end = hidden_offsets[idx], hidden_offsets[idx + 1]
                scaled = self.weights[idx].to(dtype=work_dtype) / 8.0
                ising_mat[lo_start:lo_end, hi_start:hi_end] = scaled
                ising_mat[hi_start:hi_end, lo_start:lo_end] = scaled.t()

            eff_hidden_biases = [bias.to(dtype=work_dtype) for bias in self.biases[1:]]
            eff_hidden_biases[0] = (
                eff_hidden_biases[0] + vis_aligned @ self.weights[0].to(dtype=work_dtype)
            )
            full_hidden_bias = torch.cat(eff_hidden_biases, dim=0)
            ising_bias = full_hidden_bias / 4.0 + ising_mat.sum(dim=0)[:-1]
            ising_mat[: self.num_hidden, -1] = ising_bias
            ising_mat[-1, : self.num_hidden] = ising_bias
            return ising_mat.detach().cpu().numpy()

    def _solve_ising(self, sampler, ising_mat: np.ndarray, expected_vars: int) -> torch.Tensor:
        """Invoke a Kaiwu-compatible sampler and decode binary states."""
        with kpp_caller_context():
            solution = sampler.solve(ising_mat)
        spins = np.atleast_2d(np.asarray(solution, dtype=np.float64))
        if (
            spins.ndim != 2
            or spins.shape[0] == 0
            or spins.shape[1] != expected_vars + 1
            or not np.all(np.isfinite(spins))
        ):
            raise RuntimeError(
                f"sampler must return finite spin rows of length {expected_vars + 1}, "
                f"got shape {spins.shape}"
            )
        binary = (spins[:, :-1] * spins[:, -1:] + 1.0) / 2.0
        return torch.tensor(
            binary,
            dtype=self.weights[0].dtype,
            device=self.weights[0].device,
        )

    def sample(self, sampler) -> torch.Tensor:
        """Jointly sample all visible and hidden layers ``[v, h^(1), ..., h^(L)]``.

        Args:
            sampler: Kaiwu SDK optimizer (e.g., ``SimulatedAnnealingOptimizer``
                or ``CIMOptimizer``) exposing ``solve(ising_mat)``.

        Returns:
            torch.Tensor: Sampled binary joint states of shape
            ``(num_reads, num_nodes)``.
        """
        return self._solve_ising(sampler, self._to_ising_matrix(), self.num_nodes)

    def clamped_sample(self, sampler, s_visible: torch.Tensor) -> torch.Tensor:
        """Sample hidden layers from ``p(h^(1), ..., h^(L) | v)`` via an Ising sampler.

        Args:
            sampler: Kaiwu SDK optimizer exposing ``solve(ising_mat)``.
            s_visible (torch.Tensor): Visible states of shape ``(num_visible,)``
                or ``(B, num_visible)``.

        Returns:
            torch.Tensor: Concatenated joint states ``[v, h^(1), ..., h^(L)]``
            with rows aligned to each input visible sample.
        """
        if not isinstance(s_visible, torch.Tensor):
            raise TypeError(
                f"s_visible must be a torch.Tensor, got {type(s_visible).__name__}"
            )
        if s_visible.ndim == 1:
            vis_batch = self._prepare_tensor(s_visible.unsqueeze(0), self.num_visible, "s_visible")
        else:
            vis_batch = self._prepare_tensor(s_visible, self.num_visible, "s_visible")
        if vis_batch.shape[0] == 0:
            raise ValueError("s_visible batch must be non-empty")

        chunks = []
        for idx in range(vis_batch.shape[0]):
            ising_mat = self.clamped_ising_matrix(vis_batch[idx])
            hid_samples = self._solve_ising(sampler, ising_mat, self.num_hidden)
            vis_rep = vis_batch[idx].unsqueeze(0).expand(hid_samples.shape[0], -1)
            chunks.append(torch.cat([vis_rep, hid_samples], dim=-1))
        return torch.cat(chunks, dim=0)
