# -*- coding: utf-8 -*-
"""Boltzmann Machine"""

import numpy as np
import torch

from .abstract_boltzmann_machine import AbstractBoltzmannMachine


class BoltzmannMachine(AbstractBoltzmannMachine):
    """Boltzmann Machine.

    Args:
        num_nodes (int): Total number of nodes in the model.

        quadratic_coef (torch.FloatTensor, optional): quadratic coefficent,
    shape is [num_nodes, num_nodes]

        linear_bias (torch.FloatTensor, optional): linear bias, shape is [num_nodes]

        device (torch.device, optional): Device for tensor construction.
        If ``None``, uses CPU.
    """

    def __init__(
        self,
        num_nodes: int,
        quadratic_coef: torch.FloatTensor = None,
        linear_bias: torch.FloatTensor = None,
        device=None,
    ):
        super().__init__(device=device)
        self.num_nodes = num_nodes
        self.quadratic_coef = torch.nn.Parameter(
            quadratic_coef
            if quadratic_coef is not None
            else torch.randn((self.num_nodes, self.num_nodes)).to(self.device) * 0.01
        )
        self.linear_bias = torch.nn.Parameter(
            linear_bias
            if linear_bias is not None
            else torch.zeros(self.num_nodes).to(self.device)
        )

    def symmetrized_quadratic_coef(self):
        """Quadratic coefficient"""
        quadratic_coef = self.quadratic_coef.triu(1)
        return quadratic_coef + quadratic_coef.transpose(0, 1)

    def clip_parameters(self, h_range, j_range) -> None:
        """Clip linear and quadratic bias weights in-place.

        Args:
            h_range (tuple[float, float]): Range for quadratic weights. for example, [-1, 1]
            j_range (tuple[float, float]): Range for linear weights. for example, [-1, 1]
        """
        self.get_parameter("linear_bias").data.clamp_(*h_range)
        self.get_parameter("quadratic_coef").data.clamp_(*j_range)

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

    def forward(self, s_all: torch.Tensor) -> torch.Tensor:
        """Compute the Hamiltonian.

        Args:
            s_all (torch.tensor): Tensor of shape (B, N), where B is batch size,
                N is the number of variables in the model.

        Returns:
            torch.tensor: Hamiltonian of shape (B,).
        """
        return -s_all @ self.linear_bias - 0.5 * torch.sum(
            s_all.matmul(self.symmetrized_quadratic_coef()) * s_all, dim=-1
        )

    def _to_ising_matrix(self):
        """Convert Boltzmann Machine to Ising matrix."""
        with torch.no_grad():
            linear_bias = self.linear_bias
            quadratic_coef = self.symmetrized_quadratic_coef()  # quadratic_coef
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
            quadratic_coef = self.symmetrized_quadratic_coef()
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
            q_coef = self.symmetrized_quadratic_coef()
            for _ in range(num_steps):
                # Random update order (Gibbs sampling)
                update_order = torch.randperm(self.num_nodes, device=self.device)
                for unit in update_order:
                    if unit < n_vis:
                        # Skip visible units (only sample hidden units)
                        continue
                    # Compute activation value (logit of conditional probability)
                    activation = (
                        torch.matmul(s_all, q_coef[:, unit]) + self.linear_bias[unit]
                    )
                    # Get activation probability via sigmoid
                    prob = torch.sigmoid(activation)
                    # Sample current unit state according to probability
                    s_all[:, unit] = (prob > torch.rand_like(prob)).float()
            # Return sampled states of all units
            return s_all

    def condition_sample(self, sampler, s_visible, dtype=torch.float32) -> torch.Tensor:
        """Sample from the Boltzmann Machine given some nodes.

        Args:
            sampler (kaiwu.core.Optimizer): Optimizer used for sampling from the model.
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
            solution = torch.tensor(solution, dtype=dtype, device=self.device)
            solution = torch.cat(
                [s_visible[i].unsqueeze(0).expand(solution.shape[0], -1), solution],
                dim=-1,
            )
            solutions.append(solution)
        solutions = torch.cat(solutions, dim=0)
        return solutions
