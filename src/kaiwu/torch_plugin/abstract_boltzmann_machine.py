# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0


"""Abstract base class for Boltzmann Machines."""
import torch

from kaiwu.torch_plugin.usage_stats import kpp_caller_context


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

    def _apply(self, fn, recurse=True):
        """Applies ``fn`` to parameters and keeps ``self.device`` in sync.

        ``nn.Module`` moves child modules through ``_apply`` without calling a
        child's ``.to()`` override, so a Boltzmann machine nested inside
        another module would otherwise keep a stale plain-attribute device
        after its parameters were moved. Syncing here covers every move path:
        direct ``.to()``/``.cpu()`` calls and moves initiated by parent
        modules.

        Args:
            fn: Function applied to each parameter and buffer.
            recurse: Whether to recurse into child modules.

        Returns:
            AbstractBoltzmannMachine: The moved module.
        """
        module = super()._apply(fn, recurse)
        parameter = next(self.parameters(), None)
        if parameter is not None:
            self.device = parameter.device
        return module

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
        *,
        positive_weights=None,
        negative_weights=None,
    ) -> torch.Tensor:
        """Objective function whose gradient is equivalent to the gradient of
        negative log-likelihood.

        Args:
            s_positive (torch.Tensor): Tensor of observed spins (data), shape (b1, N),
                            where b1 is batch size and N is the number of variables.
            s_negative (torch.Tensor): Tensor of spins sampled from the model, shape (b2, N),
                            where b2 is batch size and N is the number of variables.
            positive_weights (array-like, optional): Per-row weights for the data phase.
            negative_weights (array-like, optional): Per-row weights for the model phase.
                Each supplied vector must be finite and nonnegative with positive total
                weight. Weights are treated as constants and normalized per phase.

        Returns:
            torch.Tensor: Scalar difference between data and model average energy.
        """
        positive = self._phase_mean(self(s_positive), positive_weights)
        negative = self._phase_mean(self(s_negative), negative_weights)
        return positive - negative

    @staticmethod
    def _phase_mean(energies, weights):
        """Reduce one energy vector using optional observation counts."""
        if weights is None:
            return energies.mean()
        weights = torch.as_tensor(weights, device=energies.device).detach()
        if weights.shape != energies.shape or weights.ndim != 1 or weights.numel() == 0:
            raise ValueError("Phase weights must provide one weight per energy row")
        if weights.is_complex():
            raise ValueError("Phase weights must be real")
        common_dtype = torch.promote_types(energies.dtype, weights.dtype)
        weights = weights.to(dtype=torch.promote_types(common_dtype, torch.float32))
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Phase weights must be finite and nonnegative")
        scale = weights.max()
        if scale == 0:
            raise ValueError("Phase weights must have positive total weight")
        # Scale before summing to keep the reduction safe with large counts.
        weights = weights / scale
        return (energies * weights).sum() / weights.sum()

    def sample(self, sampler) -> torch.Tensor:
        """Samples from the Boltzmann Machine.

        Args:
            sampler (kaiwu.core.OptimizerBase): Optimizer used for sampling from the model.
                The sampler can be kaiwuSDK's CIM or other solvers.

        Returns:
            torch.Tensor: Spins sampled from the model.
        """
        ising_mat = self.get_ising_matrix()

        # kpp stats: set _caller_context so kaiwu @track_data decorator
        # prefixes alg_name ('sa' -> 'kpp_sa') and CIM tasks carry
        # task_source_detail.
        with kpp_caller_context():
            solution = sampler.solve(ising_mat)

        solution = (solution[:, :-1] * solution[:, [-1]] + 1) / 2
        solution = torch.FloatTensor(solution)
        solution = solution.to(self.device)

        return solution
