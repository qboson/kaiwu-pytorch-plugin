# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
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

        quadratic_coef (torch.FloatTensor, optional): quadratic coefficent,
            shape is [num_visible, num_hidden]

        linear_bias (torch.FloatTensor, optional): linear bias, shape is [num_hidden]

        device (torch.device, optional): Device to construct tensors.
    """

    def __init__(
        self,
        num_visible: int,
        num_hidden: int,
        quadratic_coef: torch.FloatTensor = None,
        linear_bias: torch.FloatTensor = None,
        device=None,
    ):
        super().__init__(device=device)
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_nodes = num_visible + num_hidden
        self.quadratic_coef = torch.nn.Parameter(
            quadratic_coef
            if quadratic_coef is not None
            else torch.randn((num_visible, num_hidden)).to(self.device) * 0.01
        )
        self.linear_bias = torch.nn.Parameter(
            linear_bias
            if linear_bias is not None
            else torch.zeros(num_hidden + num_visible).to(self.device)
        )

    @property
    def hidden_bias(self) -> torch.Tensor:
        """Return the hidden bias."""
        return self.linear_bias[self.num_visible :]

    @property
    def visible_bias(self) -> torch.Tensor:
        """Return the visible bias."""
        return self.linear_bias[: self.num_visible]

    def clip_parameters(self, h_range, j_range) -> None:
        """Clip linear and quadratic bias weights in-place.

        Args:
            h_range (tuple[float, float]): Range for quadratic weights. for example, [-1, 1]
            j_range (tuple[float, float]): Range for linear weights. for example, [-1, 1]
        """
        self.get_parameter("linear_bias").data.clamp_(*h_range)
        self.get_parameter("quadratic_coef").data.clamp_(*j_range)

    def get_hidden(
        self,
        s_visible: torch.Tensor,
        requires_grad: bool = False,
        bernoulli: bool = False,
    ) -> torch.Tensor:
        """Propagate visible spins to the hidden layer.

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
            if bernoulli:
                s_all[:, self.num_visible :] = (prob > torch.rand_like(prob)).float()
            else:
                s_all[:, self.num_visible :] = prob
            return s_all

    def get_visible(
        self, s_hidden: torch.Tensor, bernoulli: bool = False
    ) -> torch.Tensor:
        """Propagate hidden spins to the visible layer."""
        with torch.no_grad():
            s_all = torch.zeros(
                s_hidden.size(0), self.num_hidden + self.num_visible
            ).to(self.device)
            s_all[:, self.num_visible :] = s_hidden
            prob = torch.sigmoid(
                s_hidden @ self.quadratic_coef.t()
                + self.linear_bias[: self.num_visible]
            )

            if bernoulli:
                s_all[:, : self.num_visible] = (prob > torch.rand_like(prob)).float()
            else:
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
        return -s_all @ self.linear_bias - torch.sum(
            tmp * s_all[:, self.num_visible :], dim=-1
        )

    def _to_ising_matrix(self):
        """Convert the Restricted Boltzmann Machine to Ising format."""
        num_nodes = self.linear_bias.shape[-1]
        with torch.no_grad():
            ising_mat = torch.zeros((num_nodes + 1, num_nodes + 1), device=self.device)
            # Restricted Boltzmann Machine: only connections between visible and hidden layers
            ising_mat[: self.num_visible, self.num_visible : -1] = (
                self.quadratic_coef / 8
            )
            ising_mat[self.num_visible : -1, : self.num_visible] = (
                self.quadratic_coef.t() / 8
            )
            ising_bias = self.linear_bias / 4 + ising_mat.sum(dim=0)[:-1]
            ising_mat[:num_nodes, -1] = ising_bias
            ising_mat[-1, :num_nodes] = ising_bias
            return ising_mat.detach().cpu().numpy()
