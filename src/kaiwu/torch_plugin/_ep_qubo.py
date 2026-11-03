# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""K-bit QUBO mapping for equilibrium propagation.

A non-input unit with value ``s`` in ``[0, 1]`` is encoded as

``s = sum_k 2^k q_k / (2^K - 1)``, ``q_k`` in ``{0, 1}``.

Substituting that expansion into ``F_beta`` and using ``q^2 = q`` produces an
upper-triangular matrix ``Q`` whose objective ``q^T Q q`` matches ``F_beta``
up to the target-only constant ``beta/2 ||y||^2``.
"""

from __future__ import annotations

import torch


def logical_bit_count(layer_sizes, state_bits):
    """Return the number of logical bits for non-input units.

    Args:
        layer_sizes (Sequence[int]): Widths from the clamped input through the
            output layer.
        state_bits (int): Bits used to encode one unit.

    Returns:
        int: ``state_bits`` times the number of non-input units.
    """
    return int(state_bits) * sum(int(size) for size in layer_sizes[1:])


def bit_weights(state_bits, dtype, device):
    """Return the little-endian place values of one K-bit unit.

    Args:
        state_bits (int): Number of bits per unit.
        dtype (torch.dtype): Floating dtype of the returned weights.
        device (torch.device): Device of the returned weights.

    Returns:
        torch.Tensor: Shape ``(state_bits,)``, entry ``k`` is
        ``2^k / (2^K - 1)``.
    """
    levels = float((1 << int(state_bits)) - 1)
    powers = [float(1 << index) / levels for index in range(int(state_bits))]
    return torch.tensor(powers, dtype=dtype, device=device)


def decode_bits(bits, layer_sizes, state_bits):
    """Decode a logical-bit vector into one state per non-input layer.

    Args:
        bits (torch.Tensor): Flat ``{0, 1}`` vector in unit-major, little-endian
            bit order.
        layer_sizes (Sequence[int]): Widths including the clamped input.
        state_bits (int): Bits used to encode one unit.

    Returns:
        list[torch.Tensor]: States in ``[0, 1]``, output layer last.
    """
    weights = bit_weights(state_bits, bits.dtype, bits.device)
    states = []
    cursor = 0
    for width in layer_sizes[1:]:
        count = int(width) * int(state_bits)
        grouped = bits[cursor : cursor + count].reshape(int(width), int(state_bits))
        states.append(grouped @ weights)
        cursor += count
    return states


def encode_states(states, state_bits):
    """Encode layer states on the K-bit grid back into logical bits.

    Args:
        states (Sequence[torch.Tensor]): One vector per non-input layer.
        state_bits (int): Bits used to encode one unit.

    Returns:
        torch.Tensor: Flat bit vector matching :func:`decode_bits`.
    """
    levels = (1 << int(state_bits)) - 1
    chunks = []
    for state in states:
        shifts = torch.arange(int(state_bits), device=state.device)
        coded = torch.round(state * levels).to(dtype=torch.int64)
        bits = ((coded.unsqueeze(-1) >> shifts) & 1).to(dtype=state.dtype)
        chunks.append(bits.reshape(-1))
    return torch.cat(chunks)


def build_qubo(fields, self_coef, couplings, state_bits):
    """Build the upper-triangular QUBO for one equilibrium-propagation sample.

    Args:
        fields (Sequence[torch.Tensor]): Linear field of every non-input layer,
            output last. The clamped input and the nudge are already included.
        self_coef (Sequence[torch.Tensor]): Coefficient of ``1/2 ||s||^2`` on
            each of those layers. Hidden layers use 1; the output uses ``1+beta``.
        couplings (Sequence[torch.Tensor]): Weight matrices between consecutive
            non-input layers. Empty when the network has no hidden layer.
        state_bits (int): Bits used to encode one unit.

    Returns:
        torch.Tensor: Upper-triangular ``Q`` with ``q^T Q q`` equal to
        ``F_beta`` minus ``beta/2 ||y||^2``.

    Raises:
        ValueError: If the field, self-coefficient, or coupling lists disagree.
    """
    if len(fields) != len(self_coef) or len(couplings) != len(fields) - 1:
        raise ValueError("QUBO terms must align with the non-input layers")
    dtype = fields[0].dtype
    device = fields[0].device
    weights = bit_weights(state_bits, dtype, device)
    widths = [int(field.reshape(-1).shape[0]) for field in fields]
    matrix = torch.zeros(
        (int(state_bits) * sum(widths),) * 2, dtype=dtype, device=device
    )
    starts = _layer_starts(widths, state_bits)
    _add_self_terms(matrix, fields, self_coef, starts, weights, state_bits)
    _add_couplings(matrix, couplings, widths, starts, weights, state_bits)
    return matrix


def _layer_starts(widths, state_bits):
    """Return the first logical-bit index of each non-input layer."""
    starts = []
    cursor = 0
    for width in widths:
        starts.append(cursor)
        cursor += int(width) * int(state_bits)
    return starts


def _add_self_terms(matrix, fields, self_coef, starts, weights, state_bits):
    """Write the quadratic and linear self terms of every unit."""
    offdiag = torch.triu(torch.outer(weights, weights), diagonal=1)
    diagonal = 0.5 * weights * weights
    for index, (field, coef) in enumerate(zip(fields, self_coef)):
        flat_field = field.reshape(-1)
        flat_coef = coef.reshape(-1)
        for unit in range(flat_field.numel()):
            start = starts[index] + unit * int(state_bits)
            stop = start + int(state_bits)
            block = matrix[start:stop, start:stop]
            block += flat_coef[unit] * offdiag
            positions = torch.arange(int(state_bits), device=matrix.device)
            block[positions, positions] += (
                flat_coef[unit] * diagonal - flat_field[unit] * weights
            )


def _add_couplings(matrix, couplings, widths, starts, weights, state_bits):
    """Write the upper-triangular blocks between adjacent non-input layers."""
    pair = torch.outer(weights, weights)
    for index, coupling in enumerate(couplings):
        block = -torch.kron(coupling, pair)
        left = starts[index]
        right = starts[index + 1]
        left_count = int(widths[index]) * int(state_bits)
        right_count = int(widths[index + 1]) * int(state_bits)
        matrix[left : left + left_count, right : right + right_count] += block
