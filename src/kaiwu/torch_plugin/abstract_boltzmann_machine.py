# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0


"""Abstract base class for Boltzmann Machines."""
import numpy as np
import torch

from kaiwu.torch_plugin.usage_stats import kpp_caller_context


def _validate_ising_solutions(solution, num_spins):
    """Validate the complete solver batch before gauge decoding."""
    if solution is None:
        raise RuntimeError(
            "sampler.solve returned None; no Ising solutions are available. "
            "For an asynchronous sampler, complete the task before sampling."
        )
    try:
        solution = np.asarray(solution)
    except (TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError("sampler.solve returned an invalid Ising batch") from error

    if solution.ndim != 2 or solution.shape[0] == 0 or solution.shape[1] != num_spins:
        raise RuntimeError(
            "sampler.solve must return a nonempty 2D Ising batch with "
            f"{num_spins} columns (including the gauge spin); got {solution.shape}"
        )
    if solution.dtype.kind not in "biuf" or not np.all(
        (solution == -1) | (solution == 1)
    ):
        raise RuntimeError(
            "sampler.solve must return real Ising spins exactly equal to -1 or +1 "
            "in every row, including the gauge spin"
        )
    return solution


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
                The sampler can be kaiwuSDK's CIM or other solvers. Its ``solve``
                result must be a nonempty 2D batch of exact -1/+1 spins, with
                one column per Ising matrix row, including the last gauge spin.

        Returns:
            torch.Tensor: Spins sampled from the model.

        Raises:
            RuntimeError: If the sampler has no available results or returns
                an invalid Ising batch.
        """
        ising_mat = self.get_ising_matrix()

        # kpp stats: set _caller_context so kaiwu @track_data decorator
        # prefixes alg_name ('sa' -> 'kpp_sa') and CIM tasks carry
        # task_source_detail.
        with kpp_caller_context():
            solution = sampler.solve(ising_mat)

        solution = _validate_ising_solutions(solution, ising_mat.shape[0])
        solution = (solution[:, :-1] * solution[:, [-1]] + 1) / 2
        solution = torch.FloatTensor(solution)
        solution = solution.to(self.device)

        return solution
