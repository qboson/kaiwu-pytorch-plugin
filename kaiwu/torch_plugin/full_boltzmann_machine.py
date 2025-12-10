# -*- coding: utf-8 -*-
"""Boltzmann Machine"""

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize

from .restricted_boltzmann_machine import AbstractBoltzmannMachine


class BoltzmannMachineQuadraticCoef(nn.Module):
    """
    Parametrization to ensure the quadratic coefficient matrix is symmetric and has zero diagonal.
    """
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input quadratic coefficient matrix.
        Returns:
            torch.Tensor: Symmetric quadratic coefficient matrix with zero diagonal.
        """
        x_triu = x.triu(1)
        return x_triu + x_triu.transpose(0, 1)


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

    def __init__(
        self, num_nodes: int, h_range=None, j_range=None, device=None
    ):  # 对源码进行了device参数的改动，以及二次项系数的改动
        super().__init__(h_range=h_range, j_range=j_range, device=device)
        self.num_nodes = num_nodes
        self.quadratic_coef = torch.nn.Parameter(
            torch.randn((self.num_nodes, self.num_nodes)) * 0.01
        )
        parametrize.register_parametrization(
            self, "quadratic_coef", BoltzmannMachineQuadraticCoef()
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
        self.get_parameter("parametrizations.quadratic_coef.original").data.clamp_(
            *self.j_range
        )

    def forward(self, s_all: torch.Tensor) -> torch.Tensor:
        """Compute the Hamiltonian.

        Args:
            s_all (torch.tensor): Tensor of shape (B, N), where B is batch size,
                N is the number of variables in the model.

        Returns:
            torch.tensor: Hamiltonian of shape (B,).
        """
        return -s_all @ self.linear_bias - 0.5 * torch.sum(
            s_all.matmul(self.quadratic_coef) * s_all, dim=-1
        )

    def _to_ising_matrix(self):
        """Convert Boltzmann Machine to Ising matrix."""
        with torch.no_grad():
            linear_bias = self.linear_bias
            quadratic_coef = self.quadratic_coef
            print(linear_bias.size(), quadratic_coef.size())
            column_sums = torch.sum(quadratic_coef, dim=0)
            num_nodes = self.num_nodes

            ising_mat = torch.zeros(
                (num_nodes + 1, num_nodes + 1),
                device=self.device,
                dtype=linear_bias.dtype,
            )
            # Fill quadratic part
            ising_mat[:-1, :-1] = quadratic_coef / 8
            # Calculate ising_bias
            ising_bias = linear_bias / 4 + column_sums / 8
            # Fill bias part
            ising_mat[:num_nodes, -1] = ising_bias
            ising_mat[-1, :num_nodes] = ising_bias
            return ising_mat.cpu().numpy()

    def _hidden_to_ising_matrix(self, s_visible: torch.Tensor) -> np.ndarray:
        """Given visible nodes, convert the model to a submatrix in Ising format.

        Args:
            s_visible (torch.Tensor): State of the visible layer, shape (B, num_visible).

        Returns:
            np.ndarray: Submatrix in Ising format.
        """
        with torch.no_grad():
            linear_bias = self.linear_bias
            quadratic_coef = self.quadratic_coef
            n_vis = s_visible.shape[-1]
            num_nodes = self.num_nodes
            n_hid = num_nodes - n_vis
            sub_quadratic = quadratic_coef[n_vis:, n_vis:]
            sub_column_sums = torch.sum(sub_quadratic, dim=0)
            sub_quadratic_vh = quadratic_coef[n_vis:, :n_vis]
            sub_linear = sub_quadratic_vh @ s_visible + linear_bias[n_vis:]

            ising_mat = torch.zeros(
                (n_hid + 1, n_hid + 1),
                device=self.device,
                dtype=sub_linear.dtype,
            )
            ising_mat[:-1, :-1] = sub_quadratic / 8
            ising_bias = sub_linear / 4 + sub_column_sums / 4
            ising_mat[:-1, -1] = ising_bias
            ising_mat[-1, :-1] = ising_bias
            return ising_mat.cpu().numpy()

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

    def condition_sample(
        self, sampler, s_visible, dtype=torch.float32
    ) -> torch.Tensor:  # 对源码进行了dtype和device的改动
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
            solution = torch.tensor(
                solution, dtype=dtype, device=self.device
            )  # 对源码进行了dtype和device的改动
            solution = torch.cat(
                [s_visible[i].unsqueeze(0).expand(solution.shape[0], -1), solution],
                dim=-1,
            )
            solutions.append(solution)
        solutions = torch.cat(solutions, dim=0)
        return solutions
