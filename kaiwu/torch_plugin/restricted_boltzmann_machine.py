# -*- coding: utf-8 -*-
# Copyright (C) 2025-present Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Restricted Boltzmann Machine"""
import torch
from .abstract_boltzmann_machine import AbstractBoltzmannMachine


class RestrictedBoltzmannMachine(AbstractBoltzmannMachine):
    """Create a Restricted Boltzmann Machine.

    Args:
        num_visible (int): Number of visible nodes in the model.
        num_hidden (int): Number of hidden nodes in the model.
        h_range (tuple[float, float], optional): Range for linear weights.
            If ``None``, an infinite range is used.
        j_range (tuple[float, float], optional): Range for quadratic weights.
            If ``None``, an infinite range is used.
        device (torch.device, optional): Device to construct tensors.
    """

    def __init__(self, num_visible, num_hidden, h_range=None, j_range=None):
        super().__init__(h_range=h_range, j_range=j_range)
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_nodes = num_visible + num_hidden
        self.quadratic_coef = torch.nn.Parameter(
            torch.randn((num_visible, num_hidden)) * 0.01
        )
        self.linear_bias = torch.nn.Parameter(torch.zeros(num_hidden + num_visible))

    @property
    def hidden_bias(self) -> torch.Tensor:
        """Return the hidden bias."""
        return self.linear_bias[self.num_visible :]

    @property
    def visible_bias(self) -> torch.Tensor:
        """Return the visible bias."""
        return self.linear_bias[: self.num_visible]

    def clip_parameters(self) -> None:
        """Clip the linear and quadratic bias weights in place."""
        self.get_parameter("linear_bias").data.clamp_(*self.h_range)
        self.get_parameter("quadratic_coef").data.clamp_(*self.j_range)

    def get_hidden(
        self, s_visible: torch.Tensor, requires_grad: bool = False
    ) -> torch.Tensor:
        """Propagate hidden spins to the visible layer.

        Args:
            s_visible: Visible layer tensor.
            requires_grad: Whether to allow gradient backpropagation.
        """
        context = torch.enable_grad if requires_grad else torch.no_grad
        with context():
            s_all = torch.zeros(
                s_visible.size(0),
                self.num_hidden + self.num_visible,
                device=self.device,
            )
            s_all[:, : self.num_visible] = s_visible
            prob = torch.sigmoid(
                s_visible @ self.quadratic_coef + self.linear_bias[self.num_visible :]
            )
            s_all[:, self.num_visible :] = prob
            return s_all

    def get_visible(self, s_hidden: torch.Tensor) -> torch.Tensor:
        """Propagate visible spins to the hidden layer."""
        with torch.no_grad():
            s_all = torch.zeros(
                s_hidden.size(0), self.num_hidden + self.num_visible
            ).to(self.device)
            s_all[:, self.num_visible :] = s_hidden
            prob = torch.sigmoid(
                s_hidden @ self.quadratic_coef.t()
                + self.linear_bias[: self.num_visible]
            )
            s_all[:, : self.num_visible] = prob
            return s_all

    def forward(self, s_all: torch.Tensor) -> torch.Tensor:
        """Compute the Hamiltonian.

        Args:
            s_all (torch.tensor): Tensor of shape (B, N), where B is the batch size,
                and N is the number of variables in the model.

        Returns:
            torch.tensor: Hamiltonian of shape (B,).
        """
        tmp = s_all[:, : self.num_visible].matmul(self.quadratic_coef)
        return - s_all @ self.linear_bias - torch.sum(
            tmp * s_all[:, self.num_visible :], dim=-1
        )

    def _to_ising_matrix(self):
        """Convert the Restricted Boltzmann Machine to Ising format."""
        num_nodes = self.linear_bias.shape[-1]
        with torch.no_grad():
            ising_mat = torch.zeros((num_nodes + 1, num_nodes + 1), device=self.device)
            # Restricted Boltzmann Machine: only connections between visible and hidden layers
            ising_mat[: self.num_visible, self.num_visible : -1] = (
                self.quadratic_coef / 4
            )
            ising_mat[self.num_visible : -1, : self.num_visible] = (
                self.quadratic_coef.t() / 4
            )
            ising_bias = self.linear_bias / 2 + ising_mat.sum(dim=0)[:-1]
            ising_mat[:num_nodes, -1] = ising_bias / 2
            ising_mat[-1, :num_nodes] = ising_bias / 2
            return ising_mat.detach().cpu().numpy()
