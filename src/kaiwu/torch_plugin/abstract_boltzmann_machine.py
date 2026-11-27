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
            # Gaussian-Bernoulli sampling stores its dtype separately for
            # work buffers. Parent-initiated moves only pass through _apply,
            # so leaving this stale would make sample() cast correct float64
            # states back to the constructor's original float32 dtype.
            if hasattr(self, "dtype"):
                self.dtype = parameter.dtype
        return module

    def _to_model_tensor(self, value, dtype=None) -> torch.Tensor:
        """Build a tensor using the model parameter dtype and device.

        Samplers commonly return NumPy arrays. Deriving dtype and device from
        the parameters keeps sampled states compatible with the model after
        ``.double()`` or device moves instead of silently forcing float32.

        Args:
            value: Array-like sampler output or model state.
            dtype (torch.dtype, optional): Explicit dtype that overrides the
                model parameter dtype.

        Returns:
            torch.Tensor: Tensor on the model device with the requested dtype.
        """
        parameter = getattr(self, "linear_bias", None)
        if parameter is None:
            parameter = next(self.parameters(), None)
        if parameter is None:
            return torch.as_tensor(value, dtype=dtype, device=self.device)
        return torch.as_tensor(
            value,
            dtype=parameter.dtype if dtype is None else dtype,
            device=parameter.device,
        )

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

        # kpp stats: set _caller_context so kaiwu @track_data decorator
        # prefixes alg_name ('sa' -> 'kpp_sa') and CIM tasks carry
        # task_source_detail.
        with kpp_caller_context():
            solution = sampler.solve(ising_mat)

        solution = (solution[:, :-1] * solution[:, [-1]] + 1) / 2
        return self._to_model_tensor(solution)
