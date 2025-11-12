# -*- coding: utf-8 -*-
# Copyright (C) 2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Boltzmann Machine"""
import torch
import numpy as np
from .restricted_boltzmann_machine import AbstractBoltzmannMachine


class BoltzmannMachine(AbstractBoltzmannMachine):
    """Boltzmann Machine.

    Args:
        num_nodes (int): Total number of nodes in the model.
        h_range (tuple[float, float], optional): Range for linear weights.
            If ``None``, uses infinite range.
        j_range (tuple[float, float], optional): Range for quadratic weights.
            If ``None``, uses infinite range.
        device (torch.device, optional): Device for tensor construction.
            If ``None``, uses CPU.
    """

    def __init__(self, num_nodes: int, h_range=None, j_range=None):
        super().__init__(h_range=h_range, j_range=j_range)
        self.num_nodes = num_nodes
        self.quadratic_coef = torch.nn.Parameter(
            torch.randn((self.num_nodes, self.num_nodes)) * 0.01
        )
        self.linear_bias = torch.nn.Parameter(torch.zeros(self.num_nodes))

    def hidden_bias(self, num_hidden: int) -> torch.Tensor:
        """Get the hidden bias.

        Args:
            num_hidden (int): Number of hidden nodes.
        """
        num_visible = self.num_nodes - num_hidden
        return self.linear_bias[num_visible:]

    def visible_bias(self, num_visible) -> torch.Tensor:
        """Get the visible bias.
        Args:
            num_visible (int): Number of visible nodes.
        """
        return self.linear_bias[:num_visible]

    def clip_parameters(self) -> None:
        """Clip linear and quadratic bias weights in-place."""
        self.get_parameter("linear_bias").data.clamp_(*self.h_range)
        self.get_parameter("quadratic_coef").data.clamp_(*self.j_range)

    def forward(self, s_all: torch.Tensor) -> torch.Tensor:
        """Compute the Hamiltonian.

        Args:
            s_all (torch.tensor): Tensor of shape (B, N), where B is batch size,
                N is the number of variables in the model.

        Returns:
            torch.tensor: Hamiltonian of shape (B,).
        """
        tmp = s_all.matmul(self.quadratic_coef)
        return - s_all @ self.linear_bias - torch.sum(tmp * s_all, dim=-1)

    def _to_ising_matrix(self):
        """Convert Boltzmann Machine to Ising matrix."""
        with torch.no_grad():
            linear_bias = self.linear_bias.detach()
            quadratic_coef = self.quadratic_coef.detach()
            num_nodes = self.linear_bias.shape[-1]
            # Use torch for all calculations
            ising_mat = torch.zeros(
                (num_nodes + 1, num_nodes + 1),
                device=self.device,
                dtype=linear_bias.dtype,
            )

            # Fill quadratic part
            ising_mat[:-1, :-1] = quadratic_coef / 4
            # Calculate ising_bias
            diag_elements = torch.diag(ising_mat)[:-1]
            column_sums = torch.sum(ising_mat, dim=0)[:-1]
            ising_bias = linear_bias / 2 + diag_elements + column_sums
            # Fill bias part
            ising_mat[:num_nodes, -1] = ising_bias / 2
            ising_mat[-1, :num_nodes] = ising_bias / 2
            # Set diagonal to zero
            ising_mat.fill_diagonal_(0)
            return ising_mat.cpu().numpy()

    def _hidden_to_ising_matrix(self, s_visible: torch.Tensor) -> np.ndarray:
        """Given visible nodes, convert the model to a submatrix in Ising format.

        Args:
            s_visible (torch.Tensor): State of the visible layer, shape (B, num_visible).

        Returns:
            np.ndarray: Submatrix in Ising format.
        """
        with torch.no_grad():
            n_vis = s_visible.shape[-1]
            sub_quadratic = self.quadratic_coef[n_vis:, n_vis:]
            sub_quadratic_vh = (
                self.quadratic_coef[n_vis:, :n_vis]
                + self.quadratic_coef[:n_vis, n_vis:].t()
            )

            sub_linear = sub_quadratic_vh @ s_visible + self.linear_bias[n_vis:]
            ising_mat = torch.zeros(
                (sub_quadratic.size(0) + 1, sub_quadratic.size(0) + 1),
                device=self.device,
                dtype=sub_quadratic.dtype,
            )
            ising_mat[:-1, :-1] = sub_quadratic / 4
            ising_bias = (
                sub_linear / 2
                + torch.diag(ising_mat)[:-1]
                + torch.sum(ising_mat, dim=0)[:-1]
            )
            ising_mat[:-1, -1] = ising_bias / 2
            ising_mat[-1, :-1] = ising_bias / 2
            ising_mat.fill_diagonal_(0)
            return ising_mat.detach().cpu().numpy()

    def gibbs_sample(
        self, num_steps: int = 100, s_visible: torch.Tensor = None, num_sample=None
    ) -> torch.Tensor:
        """Sample from the Boltzmann Machine.

        Args:
            num_steps (int): Number of Gibbs sampling steps.
            s_visible (torch.Tensor, optional): State of the visible layer,
                shape (B, num_visible). If ``None``, randomly initialize visible layer.
            num_sample (int, optional): Number of samples.
                If ``None``, uses batch size of s_visible.
        """
        with torch.no_grad():
            # Initialization: If neither visible unit state nor sample number is provided,
            # raise error
            if s_visible is None and num_sample is None:
                raise ValueError("Either s_visible or num_sample must be provided.")
            # If sample number is not specified, use batch size of visible units
            num_sample = s_visible.size(0) if num_sample is None else num_sample
            if s_visible is not None:
                # Initialize all units (visible + hidden) with Bernoulli(0.5)
                s_all = torch.bernoulli(
                    torch.full(
                        (s_visible.size(0), self.num_nodes), 0.5, device=self.device
                    )
                )
                # Replace visible part with given visible unit state
                s_all[:, : s_visible.size(1)] = s_visible.clone()
            else:
                # If no visible units, initialize all randomly
                s_all = torch.bernoulli(
                    torch.full((num_sample, self.num_nodes), 0.5, device=self.device)
                )

            # Number of visible units
            n_vis = s_visible.shape[-1] if s_visible is not None else 0
            for _ in range(num_steps):
                # Random update order (Gibbs sampling)
                update_order = torch.randperm(self.num_nodes, device=self.device)
                for unit in update_order:
                    if unit < n_vis:
                        # Skip visible units (only sample hidden units)
                        continue
                    # Compute activation value (logit of conditional probability)
                    activation = (
                        torch.matmul(s_all, self.quadratic_coef[:, unit])
                        + self.linear_bias[unit]
                    )
                    # Get activation probability via sigmoid
                    prob = torch.sigmoid(activation)
                    # Sample current unit state according to probability
                    s_all[:, unit] = (prob > torch.rand_like(prob)).float()
            # Return sampled states of all units
            return s_all

    def condition_sample(self, sampler, s_visible) -> torch.Tensor:
        """Sample from the Boltzmann Machine given some nodes.

        Args:
            sampler: Optimizer used for sampling from the model.
            s_visible: State of the visible layer.

        Returns:
            torch.Tensor: Spins sampled from the model
                (shape determined by ``sampler`` and ``sample_params``).
        """
        solutions = []
        for i in range(s_visible.size(0)):
            ising_mat = self._hidden_to_ising_matrix(s_visible[i])
            solution = sampler.solve(ising_mat)
            solution = (solution[:, :-1] + 1) / 2
            solution = torch.cat(
                [
                    s_visible[i].unsqueeze(0).expand(solution.shape[0], -1),
                    torch.FloatTensor(solution),
                ],
                dim=-1,
            )
            solutions.append(solution)
        solutions = torch.cat(solutions, dim=0)
        solutions = torch.FloatTensor(solutions)
        solutions = solutions.to(self.device)
        return solutions
