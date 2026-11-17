# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Equilibrium propagation on layered ``+-1`` spins solved by Kaiwu optimizers.

Every non-input unit is a spin in ``{-1, +1}``. Layer ``l`` drives its spins
from the previous layer (the clamped input for the first layer) through a
linear map ``D_l``, so the free energy is

``E = -sum_l (s^l)^T (D_l(s^{l-1}) + b^l)``.

The nudged phase minimises ``E + beta * 1/2 ||s^L - y||^2`` with ``y`` in
``{-1, +1}``. For spins that cost equals ``n_out - y^T s^L``, so both phases
are Ising problems. The model writes the Ising matrix directly and hands it to
a caller-supplied Kaiwu optimizer, as the Boltzmann machines do.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .usage_stats import kpp_caller_context


class EPLinear(nn.Module):
    """Fully connected layer: ``D(s) = s @ weight``.

    Args:
        in_features (int): Width of the previous layer.
        out_features (int): Number of spins in this layer.
        device (torch.device, optional): Parameter device. Defaults to CPU.
        dtype (torch.dtype): Parameter dtype.
    """

    def __init__(self, in_features, out_features, device=None, dtype=torch.float32):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        limit = 1.0 / math.sqrt(self.in_features)
        self.weight = nn.Parameter(
            torch.empty(self.in_features, self.out_features, dtype=dtype, device=device)
            .uniform_(-limit, limit)
        )
        self.bias = nn.Parameter(torch.zeros(self.out_features, dtype=dtype, device=device))

    def drive(self, previous):
        """Return ``previous @ weight`` for rows of shape ``(B, in_features)``."""
        return previous @ self.weight

    def forward(self, previous):
        """Compute the drive. Same as :meth:`drive`."""
        return self.drive(previous)

    def coupling(self):
        """Return the ``(in_features, out_features)`` matrix of :meth:`drive`."""
        return self.weight

    def bias_vector(self):
        """Return the bias of every spin in this layer."""
        return self.bias


class EPConv2d(nn.Module):
    """Convolutional layer over a ``(channels, height, width)`` previous layer.

    Spins are laid out like the flattened output of ``F.conv2d``. Each output
    channel has one bias shared by all of its positions.

    Args:
        in_shape (tuple[int, int, int]): ``(channels, height, width)`` of the
            previous layer.
        out_channels (int): Number of output channels.
        kernel_size (int): Side of the square kernel.
        stride (int): Convolution stride.
        padding (int): Zero padding on each side.
        device (torch.device, optional): Parameter device. Defaults to CPU.
        dtype (torch.dtype): Parameter dtype.

    Raises:
        ValueError: If the kernel does not fit the padded input.
    """

    def __init__(
        self,
        in_shape,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.in_shape = tuple(int(size) for size in in_shape)
        self.stride = int(stride)
        self.padding = int(padding)
        channels, height, width = self.in_shape
        out_height = (height + 2 * self.padding - int(kernel_size)) // self.stride + 1
        out_width = (width + 2 * self.padding - int(kernel_size)) // self.stride + 1
        if out_height <= 0 or out_width <= 0:
            raise ValueError(f"kernel {kernel_size} does not fit input {self.in_shape}")
        self.out_shape = (int(out_channels), out_height, out_width)
        self.in_features = channels * height * width
        self.out_features = int(out_channels) * out_height * out_width
        limit = 1.0 / math.sqrt(channels * int(kernel_size) ** 2)
        self.weight = nn.Parameter(
            torch.empty(
                int(out_channels),
                channels,
                int(kernel_size),
                int(kernel_size),
                dtype=dtype,
                device=device,
            ).uniform_(-limit, limit)
        )
        self.bias = nn.Parameter(torch.zeros(int(out_channels), dtype=dtype, device=device))

    def drive(self, previous):
        """Convolve rows of shape ``(B, in_features)`` and flatten the result."""
        images = previous.reshape(-1, *self.in_shape)
        output = F.conv2d(images, self.weight, stride=self.stride, padding=self.padding)
        return output.reshape(images.shape[0], self.out_features)

    def forward(self, previous):
        """Compute the drive. Same as :meth:`drive`."""
        return self.drive(previous)

    def coupling(self):
        """Return the ``(in_features, out_features)`` matrix of :meth:`drive`.

        The convolution is linear, so applying it to the identity basis gives
        its dense matrix.
        """
        basis = torch.eye(self.in_features, dtype=self.weight.dtype, device=self.weight.device)
        return self.drive(basis)

    def bias_vector(self):
        """Return the bias of every spin, one value per output position."""
        positions = self.out_shape[1] * self.out_shape[2]
        return self.bias.repeat_interleave(positions)


class EquilibriumPropagation(nn.Module):
    """Equilibrium-propagation network built from a stack of layers.

    Args:
        layers (Sequence[nn.Module], optional): :class:`EPLinear` or
            :class:`EPConv2d` layers from the input through the output.
        layer_sizes (Sequence[int], optional): Widths from the input through the
            output. Builds one :class:`EPLinear` per pair when ``layers`` is not
            given.
        device (torch.device, optional): Device of layers built from
            ``layer_sizes``.
        dtype (torch.dtype): Dtype of layers built from ``layer_sizes``.

    Raises:
        ValueError: If neither or both of ``layers`` and ``layer_sizes`` are
            given, or if adjacent layers disagree on their width.
    """

    def __init__(self, layers=None, *, layer_sizes=None, device=None, dtype=torch.float32):
        super().__init__()
        if (layers is None) == (layer_sizes is None):
            raise ValueError("pass exactly one of layers and layer_sizes")
        if layers is None:
            sizes = tuple(int(size) for size in layer_sizes)
            if len(sizes) < 2 or any(size <= 0 for size in sizes):
                raise ValueError(f"layer_sizes needs at least two positive widths, got {sizes}")
            layers = [
                EPLinear(fan_in, fan_out, device=device, dtype=dtype)
                for fan_in, fan_out in zip(sizes[:-1], sizes[1:])
            ]
        self.layers = nn.ModuleList(layers)
        if len(self.layers) == 0:
            raise ValueError("the network needs at least one layer")
        for index in range(1, len(self.layers)):
            if self.layers[index].in_features != self.layers[index - 1].out_features:
                raise ValueError(
                    f"layer {index} expects {self.layers[index].in_features} inputs, "
                    f"but layer {index - 1} has {self.layers[index - 1].out_features} spins"
                )
        self.layer_sizes = (self.layers[0].in_features,) + tuple(
            layer.out_features for layer in self.layers
        )

    def energy(self, inputs, states):
        """Return ``E`` for each row.

        Args:
            inputs (torch.Tensor): Clamped inputs of shape ``(B, n_in)``.
            states (Sequence[torch.Tensor]): One ``(B, n_l)`` tensor of ``+-1``
                spins per layer, output last.

        Returns:
            torch.Tensor: Energies of shape ``(B,)``.
        """
        total = inputs.new_zeros(inputs.shape[0])
        previous = inputs
        for layer, state in zip(self.layers, states):
            total = total - (state * (layer.drive(previous) + layer.bias_vector())).sum(-1)
            previous = state
        return total

    def forward(self, inputs, states):
        """Compute the free energy. Same as :meth:`energy`."""
        return self.energy(inputs, states)

    def ising_matrix(self, inputs, target=None, beta=0.0):
        """Return the Ising matrix of one sample, auxiliary spin last.

        For spins ``s`` and ``z = [s, 1]``, ``-z^T J z`` equals
        ``E + beta * 1/2 ||s^L - y||^2`` up to a constant that does not depend
        on ``s``.

        Args:
            inputs (torch.Tensor): One clamped input of shape ``(n_in,)``.
            target (torch.Tensor, optional): ``+-1`` output target of shape
                ``(n_out,)``. Required when ``beta > 0``.
            beta (float): Nudging coefficient.

        Returns:
            np.ndarray: Symmetric ``float64`` matrix of shape ``(n + 1, n + 1)``,
            ``n`` the number of spins.

        Raises:
            ValueError: If ``beta > 0`` and ``target`` is missing.
        """
        beta = float(beta)
        if beta > 0.0 and target is None:
            raise ValueError("the nudged phase requires a target")
        widths = self.layer_sizes[1:]
        starts = np.cumsum((0,) + widths).tolist()
        size = starts[-1]
        with torch.no_grad():
            matrix = torch.zeros((size + 1, size + 1), dtype=torch.float64)
            last = len(self.layers) - 1
            for index, layer in enumerate(self.layers):
                field = layer.bias_vector().detach().to(torch.float64)
                if index == 0:
                    drive = layer.drive(inputs.reshape(1, -1).to(layer.weight.dtype))
                    field = field + drive.reshape(-1).to(torch.float64)
                else:
                    block = 0.5 * layer.coupling().detach().to(torch.float64).cpu()
                    rows = slice(starts[index - 1], starts[index])
                    cols = slice(starts[index], starts[index + 1])
                    matrix[rows, cols] = block
                    matrix[cols, rows] = block.T
                if index == last and beta > 0.0:
                    field = field + beta * target.detach().to(torch.float64).reshape(-1)
                cols = slice(starts[index], starts[index + 1])
                matrix[cols, size] = 0.5 * field.cpu()
                matrix[size, cols] = 0.5 * field.cpu()
        return matrix.numpy()

    def equilibrate(self, inputs, sampler, target=None, beta=0.0):
        """Relax each row with ``sampler`` and return the lowest-energy spins.

        Args:
            inputs (torch.Tensor): Clamped inputs of shape ``(B, n_in)``.
            sampler: Kaiwu optimizer whose ``solve(ising_mat)`` returns ``+-1``
                rows with the auxiliary spin last.
            target (torch.Tensor, optional): ``+-1`` targets of shape
                ``(B, n_out)``. Required when ``beta > 0``.
            beta (float): Nudging coefficient.

        Returns:
            list[torch.Tensor]: One ``(B, n_l)`` tensor of ``+-1`` spins per
            layer, output last.

        Raises:
            ValueError: If ``beta > 0`` and ``target`` is missing.
            RuntimeError: If ``sampler.solve`` does not return spin rows of the
                Ising matrix size.
        """
        rows = []
        for index in range(inputs.shape[0]):
            row_target = None if target is None else target[index]
            ising_mat = self.ising_matrix(inputs[index], row_target, beta)
            with kpp_caller_context():
                solution = sampler.solve(ising_mat)
            spins = np.atleast_2d(np.asarray(solution, dtype=np.float64))
            if spins.shape[0] == 0 or spins.shape[1] != ising_mat.shape[0]:
                raise RuntimeError(
                    f"sampler must return spin rows of length {ising_mat.shape[0]}, "
                    f"got shape {spins.shape}"
                )
            spins = spins * spins[:, -1:]
            hamiltonian = -np.einsum("bi,ij,bj->b", spins, ising_mat, spins)
            rows.append(spins[int(np.argmin(hamiltonian)), :-1])
        parameter = self.layers[0].weight
        states = torch.as_tensor(np.stack(rows), dtype=parameter.dtype, device=parameter.device)
        return list(torch.split(states, list(self.layer_sizes[1:]), dim=1))

    def objective(self, inputs, free_states, nudged_states, beta):
        """Return the EP loss ``mean((E_nudged - E_free) / beta)``.

        Its gradient is the free-minus-nudged correlation difference, so
        minimising it with any PyTorch optimizer applies the EP update.

        Args:
            inputs (torch.Tensor): Clamped inputs of shape ``(B, n_in)``.
            free_states (Sequence[torch.Tensor]): States from ``beta == 0``.
            nudged_states (Sequence[torch.Tensor]): States from ``beta``.
            beta (float): Strictly positive nudging coefficient.

        Returns:
            torch.Tensor: Scalar loss.

        Raises:
            ValueError: If ``beta`` is not strictly positive.
        """
        if float(beta) <= 0.0:
            raise ValueError("objective requires a strictly positive beta")
        free = [state.detach() for state in free_states]
        nudged = [state.detach() for state in nudged_states]
        return ((self.energy(inputs, nudged) - self.energy(inputs, free)) / float(beta)).mean()
