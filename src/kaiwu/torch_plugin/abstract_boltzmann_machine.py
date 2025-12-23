# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0


"""Abstract base class for Boltzmann Machines."""
import torch


class AbstractBoltzmannMachine(torch.nn.Module):
    """Abstract base class for Boltzmann Machines.

    Args:
        device (torch.device, optional): Device for tensor construction.
    """

    def __init__(self, device=None) -> None:
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def to(self, device=..., dtype=..., non_blocking=...):
        """Moves the model to the specified device.

        Args:
            device: Target device.
            dtype: Target data type.
            non_blocking: Whether the operation should be non-blocking.

        Returns:
            AbstractBoltzmannMachine: The model on the target device.
        """
        self.device = device
        return super().to(device)

    def forward(self, s_all: torch.Tensor) -> torch.Tensor:
        """Computes the Hamiltonian.

        Args:
            s_all (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Hamiltonian.
        """

    def get_ising_matrix(self):
        """Converts the model to Ising format.

        Returns:
            torch.Tensor: Ising matrix.
        """
        return self._to_ising_matrix()

    def _to_ising_matrix(self):
        """Converts the model to Ising format.

        Returns:
            torch.Tensor: Ising matrix.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Subclasses must implement _ising method")

    def objective(
        self,
        s_positive: torch.Tensor,
        s_negative: torch.Tensor,
    ) -> torch.Tensor:
        """Objective function whose gradient is equivalent to the gradient of
        negative log-likelihood.

        Args:
            s_positive (torch.Tensor): Tensor of observed spins (data), shape (b1, N),
                            where b1 is batch size and N is the number of variables.
            s_negative (torch.Tensor): Tensor of spins sampled from the model, shape (b2, N),
                            where b2 is batch size and N is the number of variables.

        Returns:
            torch.Tensor: Scalar difference between data and model average energy.
        """
        return self(s_positive).mean() - self(s_negative).mean()

    def sample(self, sampler) -> torch.Tensor:
        """Samples from the Boltzmann Machine.

        Args:
            sampler (kaiwu.core.OptimizerBase): Optimizer used for sampling from the model.
                The sampler can be kaiwuSDK's CIM or other solvers.

        Returns:
            torch.Tensor: Spins sampled from the model.
        """
        ising_mat = self.get_ising_matrix()
        solution = sampler.solve(ising_mat)
        solution = (solution[:, :-1] + 1) / 2
        solution = torch.FloatTensor(solution)
        solution = solution.to(self.device)
        return solution
