# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Equilibrium propagation energy and deterministic relaxations.

The free energy of a layered network with states in ``[0, 1]`` is

``E = 1/2 sum_l ||s^l||^2 - sum_l (s^l)^T W^l s^{l+1} - sum_l (b^l)^T s^l``.

The nudged objective adds ``beta * 1/2 ||s^L - y||^2``. This module builds the
matching K-bit QUBO. The default relaxation sends that QUBO through Kaiwu 1.3.1
simulated annealing. Exhaustive and continuous backends remain for exact checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from ._ep_dynamics import ContinuousDynamics
from ._ep_qubo import build_qubo, decode_bits, encode_states, logical_bit_count
from ._ep_solver import ExhaustiveBackend, SimulatedAnnealingBackend
from ._ep_training import predict as run_predict
from ._ep_training import train_batch as run_train_batch


@dataclass(frozen=True)
class EPConfig:
    """Configuration shared by the energy, the QUBO, and both relaxations.

    Args:
        layer_sizes (tuple[int, ...]): Widths from the clamped input through
            the output. At least two positive widths are required.
        state_bits (int): Bits used to encode each non-input unit.
        beta (float): Default nudging coefficient.
        exhaustive_max_bits (int): Largest QUBO the exhaustive backend enumerates.
        step_size (float): Initial step of the continuous relaxation.
        max_steps (int): Maximum accepted continuous updates.
        seed (int): Seed of the parameter initialisation and the default anneal.
        initial_temperature (float): Kaiwu annealing start temperature.
        alpha (float): Kaiwu geometric cooling factor.
        cutoff_temperature (float): Kaiwu annealing stop temperature.
        iterations_per_t (int): Kaiwu sweeps at each temperature.
        candidate_count (int): Kaiwu candidate pool size.
    """

    layer_sizes: tuple
    state_bits: int = 1
    beta: float = 0.0
    exhaustive_max_bits: int = 16
    step_size: float = 0.05
    max_steps: int = 40
    seed: int = 0
    initial_temperature: float = 1.25
    alpha: float = 0.95
    cutoff_temperature: float = 0.001
    iterations_per_t: int = 10
    candidate_count: int = 8

    def __post_init__(self):
        sizes = tuple(int(size) for size in self.layer_sizes)
        object.__setattr__(self, "layer_sizes", sizes)
        object.__setattr__(self, "state_bits", int(self.state_bits))
        if len(sizes) < 2 or any(size <= 0 for size in sizes):
            raise ValueError(
                f"layer_sizes must contain at least two positive widths, got {sizes}"
            )
        if self.state_bits < 1:
            raise ValueError(f"state_bits must be positive, got {self.state_bits}")
        if int(self.exhaustive_max_bits) < 1:
            raise ValueError("exhaustive_max_bits must be positive")

    def to_dict(self):
        """Return a JSON-friendly copy of this configuration."""
        payload = asdict(self)
        payload["layer_sizes"] = list(self.layer_sizes)
        return payload

    @classmethod
    def from_dict(cls, payload):
        """Rebuild a configuration from :meth:`to_dict`."""
        data = dict(payload)
        data["layer_sizes"] = tuple(data["layer_sizes"])
        return cls(**data)


class EquilibriumPropagation(nn.Module):
    """Layered equilibrium-propagation energy with a K-bit QUBO mapping.

    ``weights[l]`` has shape ``(n_l, n_{l+1})`` and ``biases[l]`` has shape
    ``(n_{l+1},)``. The input layer is clamped, so it has no bias.

    Args:
        config (EPConfig): Network widths and relaxation settings.
        dtype (torch.dtype): Parameter dtype. ``float64`` keeps the QUBO
            identity inside ``1e-9``.
        device (torch.device, optional): Parameter device. Defaults to CPU.
    """

    def __init__(self, config, dtype=torch.float64, device=None):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.layer_sizes = config.layer_sizes
        generator = torch.Generator()
        generator.manual_seed(int(config.seed))
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for fan_in, fan_out in zip(config.layer_sizes[:-1], config.layer_sizes[1:]):
            weight = torch.randn((fan_in, fan_out), generator=generator, dtype=dtype)
            self.weights.append(nn.Parameter(weight * 0.01))
            self.biases.append(nn.Parameter(torch.zeros(fan_out, dtype=dtype)))
        if device is not None:
            self.to(device)

    def energy(self, inputs, states):
        """Return ``E`` for one clamped input and its non-input states.

        Args:
            inputs (torch.Tensor): Clamped input of shape ``(n_in,)``.
            states (Sequence[torch.Tensor]): One state vector per non-input
                layer, output last.

        Returns:
            torch.Tensor: Scalar free energy.
        """
        total = inputs.new_zeros(())
        previous = inputs
        for index, state in enumerate(states):
            total = total + 0.5 * torch.dot(state, state)
            total = total - torch.dot(self.biases[index], state)
            total = total - torch.dot(previous, self.weights[index] @ state)
            previous = state
        return total

    def cost(self, output, target):
        """Return ``1/2 ||output - target||^2``."""
        delta = output - target
        return 0.5 * torch.dot(delta, delta)

    def total_energy(self, inputs, states, target, beta):
        """Return ``E + beta * C``. ``beta == 0`` ignores the target."""
        value = self.energy(inputs, states)
        if float(beta) != 0.0:
            value = value + float(beta) * self.cost(states[-1], target)
        return value

    def forward(self, inputs, states):
        """Compute the free energy. Matches :meth:`energy`."""
        return self.energy(inputs, states)

    def qubo(self, inputs, target, beta):
        """Return ``(Q, offset)`` for one sample.

        Args:
            inputs (torch.Tensor): Clamped input.
            target (torch.Tensor): Output target. Ignored when ``beta`` is 0,
                but still required so the matrix width stays fixed.
            beta (float): Nudging coefficient.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Upper-triangular ``Q`` and the
            scalar ``beta/2 ||target||^2``. For a decoded state ``s(q)``,
            ``q^T Q q + offset`` equals :meth:`total_energy`.
        """
        beta = float(beta)
        fields, coefs, couplings = self._qubo_terms(inputs, target, beta)
        matrix = build_qubo(fields, coefs, couplings, self.config.state_bits)
        offset = 0.5 * beta * torch.dot(target, target)
        return matrix, offset

    def equilibrate(
        self,
        inputs,
        target=None,
        beta=0.0,
        backend="sa",
        init_solution=None,
        seed=None,
    ):
        """Relax one sample. The default backend is Kaiwu simulated annealing.

        Args:
            inputs (torch.Tensor): Clamped input.
            target (torch.Tensor, optional): Required when ``beta > 0``.
            beta (float): Nudging coefficient.
            backend (str): ``sa``, ``exhaustive``, or ``continuous``.
            init_solution (np.ndarray, optional): Warm start for ``sa`` only,
                length ``n_bits + 1``, entries ``-1`` or ``+1``.
            seed (int, optional): Anneal seed. Defaults to ``config.seed``.

        Returns:
            dict: ``states``, recomputed ``energy``, ``energy_trace``,
            ``stop_reason``, ``n_bits``, and ``init_solution``.

        Raises:
            ValueError: If a positive nudge has no target, or the backend name
                is unknown.
        """
        beta = float(beta)
        if beta > 0.0 and target is None:
            raise ValueError("the nudged phase requires a target")
        features = self._as_vector(inputs, self.layer_sizes[0])
        labels = self._target_or_zeros(target)
        anneal_seed = self.config.seed if seed is None else int(seed)
        if backend == "sa":
            return self._equilibrate_sa(features, labels, beta, init_solution, anneal_seed)
        if backend == "exhaustive":
            return self._equilibrate_exhaustive(features, labels, beta)
        if backend == "continuous":
            return self._equilibrate_continuous(features, labels, beta)
        raise ValueError(f"unknown backend {backend!r}")

    def train_batch(self, inputs, targets, optimizer, **kwargs):
        """Solve a paired batch and take one optimizer step.

        Args:
            inputs (torch.Tensor): Batch of clamped inputs, shape ``(batch, n_in)``.
            targets (torch.Tensor): One-hot targets in ``[0, 1]``, shape
                ``(batch, n_out)``.
            optimizer (torch.optim.Optimizer): Minimizer that consumes ``.grad``.
            **kwargs: Forwarded to :func:`kaiwu.torch_plugin._ep_training.train_batch`.

        Returns:
            dict: Batch size, loss, free-phase cost, accuracy, zero-update rate
            and solver-call count.
        """
        return run_train_batch(self, inputs, targets, optimizer, **kwargs)

    def predict(self, inputs, backend="sa", seed=None):
        """Classify a batch from the free-phase output state.

        Args:
            inputs (torch.Tensor): Batch of clamped inputs.
            backend (str): Relaxation backend.
            seed (int, optional): Base seed. Defaults to ``config.seed``.

        Returns:
            torch.Tensor: Class indices of shape ``(batch,)`` and dtype ``long``.
        """
        return run_predict(self, inputs, backend=backend, seed=seed)

    def _equilibrate_sa(self, features, labels, beta, init_solution, anneal_seed):
        """Anneal the K-bit QUBO with Kaiwu and decode the best bits."""
        matrix, offset = self.qubo(features, labels, beta)
        backend = SimulatedAnnealingBackend(
            initial_temperature=self.config.initial_temperature,
            alpha=self.config.alpha,
            cutoff_temperature=self.config.cutoff_temperature,
            iterations_per_t=self.config.iterations_per_t,
            candidate_count=self.config.candidate_count,
            seed=int(anneal_seed),
        )
        bits, objective = backend.minimize(matrix, init_solution)
        states = decode_bits(bits, self.layer_sizes, self.config.state_bits)
        energy = self.total_energy(features, states, labels, beta)
        return self._finish(
            states,
            energy,
            objective + offset,
            [float(energy.detach())],
            "budget_exhausted",
        )

    def _equilibrate_exhaustive(self, features, labels, beta):
        """Pick an exact QUBO minimiser and decode it."""
        matrix, offset = self.qubo(features, labels, beta)
        backend = ExhaustiveBackend(self.config.exhaustive_max_bits)
        bits, objective = backend.minimize(matrix)
        states = decode_bits(bits, self.layer_sizes, self.config.state_bits)
        energy = self.total_energy(features, states, labels, beta)
        return self._finish(
            states, energy, objective + offset, [float(energy.detach())], "exhaustive"
        )

    def _equilibrate_continuous(self, features, labels, beta):
        """Relax ``F_beta`` with projected gradient descent."""
        dynamics = ContinuousDynamics(self.config.step_size, self.config.max_steps)

        def energy_fn(states):
            return self.total_energy(features, states, labels, beta)

        states, trace, stop_reason = dynamics.relax(
            energy_fn, self.layer_sizes[1:], self.dtype, features.device
        )
        return self._finish(states, energy_fn(states).detach(), None, trace, stop_reason)

    def _finish(self, states, energy, qubo_objective, trace, stop_reason):
        """Attach the gauge-fixed spin row a nudged phase can warm-start from."""
        return {
            "states": states,
            "energy": energy,
            "qubo_objective": qubo_objective,
            "energy_trace": trace,
            "stop_reason": stop_reason,
            "n_bits": logical_bit_count(self.layer_sizes, self.config.state_bits),
            "init_solution": self._spin_row(states),
        }

    def _spin_row(self, states):
        """Encode states as ``2q - 1`` followed by an auxiliary spin of ``+1``."""
        bits = encode_states(states, self.config.state_bits).detach().cpu().numpy()
        spins = 2.0 * np.asarray(bits, dtype=np.float64) - 1.0
        return np.concatenate([spins, np.ones(1, dtype=np.float64)])

    def _qubo_terms(self, inputs, target, beta):
        """Fold the clamped input and the nudge into per-layer QUBO fields."""
        fields = []
        coefs = []
        last = len(self.biases) - 1
        for index, bias in enumerate(self.biases):
            field = bias + (self.weights[0].transpose(0, 1) @ inputs if index == 0 else 0.0)
            coef = torch.ones_like(bias)
            if index == last:
                field = field + beta * target
                coef = coef + beta
            fields.append(field)
            coefs.append(coef)
        return fields, coefs, list(self.weights[1:])

    def _as_vector(self, value, width):
        """Convert one sample to a flat parameter-dtype vector."""
        vector = torch.as_tensor(value, dtype=self.dtype, device=self.biases[0].device)
        return vector.reshape(int(width))

    def _target_or_zeros(self, target):
        """Return the target vector, or zeros when the phase is free."""
        if target is None:
            return torch.zeros(
                self.layer_sizes[-1], dtype=self.dtype, device=self.biases[0].device
            )
        return self._as_vector(target, self.layer_sizes[-1])
