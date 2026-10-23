# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Projected gradient relaxation for continuous equilibrium propagation."""

from __future__ import annotations

import torch


class ContinuousDynamics:  # pylint: disable=too-few-public-methods
    """Minimise ``F_beta`` inside the unit box by projected gradient descent.

    A step that would raise the energy is shortened until it does not. The
    recorded trace is therefore non-increasing up to a tiny numerical tolerance.

    Args:
        step_size (float): Initial gradient step.
        max_steps (int): Maximum number of accepted updates.
        tolerance (float): Stop when the state gradient norm falls to this value.
    """

    def __init__(self, step_size=0.05, max_steps=40, tolerance=1e-6):
        self.step_size = float(step_size)
        self.max_steps = int(max_steps)
        self.tolerance = float(tolerance)

    def relax(self, energy_fn, widths, dtype, device):
        """Relax one sample from the centre of the unit box.

        Args:
            energy_fn (Callable): Maps a list of state vectors to a scalar
                ``F_beta``.
            widths (Sequence[int]): Width of each non-input layer.
            dtype (torch.dtype): State dtype.
            device (torch.device): State device.

        Returns:
            tuple: ``(states, energy_trace, stop_reason)``. ``stop_reason`` is
            ``converged``, ``max_steps``, or ``energy_increase``.
        """
        states = [
            torch.full((int(width),), 0.5, dtype=dtype, device=device) for width in widths
        ]
        value = energy_fn(states).detach()
        trace = [float(value)]
        stop_reason = "max_steps"
        for _ in range(self.max_steps):
            leaves = [state.detach().requires_grad_(True) for state in states]
            grads = torch.autograd.grad(energy_fn(leaves), leaves)
            if _grad_norm(grads) <= self.tolerance:
                stop_reason = "converged"
                break
            accepted, accepted_value = self._accept_step(energy_fn, leaves, grads, value)
            if accepted is None:
                stop_reason = "energy_increase"
                break
            states = accepted
            value = accepted_value.detach()
            trace.append(float(value))
        return states, trace, stop_reason

    def _accept_step(self, energy_fn, states, grads, value):
        """Shorten the step until the projected update does not raise energy."""
        step = self.step_size
        reference = float(value)
        tolerance = 1e-12 * max(1.0, abs(reference))
        for _ in range(12):
            candidate = [
                torch.clamp(state - step * grad, 0.0, 1.0).detach()
                for state, grad in zip(states, grads)
            ]
            candidate_value = energy_fn(candidate).detach()
            if float(candidate_value) <= reference + tolerance:
                return candidate, candidate_value
            step *= 0.5
        return None, None


def _grad_norm(grads):
    """Return the Euclidean norm of a sequence of gradient tensors."""
    total = grads[0].new_zeros(())
    for grad in grads:
        total = total + torch.sum(grad * grad)
    return float(torch.sqrt(total))
