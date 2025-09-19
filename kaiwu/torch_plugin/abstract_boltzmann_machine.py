# -*- coding: utf-8 -*-
"""Abstract base class for Boltzmann Machines."""
import torch


def clip_parameters_hook(module, *args):  # pylint:disable=unused-argument
    """Hook function for automatically clipping parameters."""
    module.clip_parameters()


class AbstractBoltzmannMachine(torch.nn.Module):
    """Abstract base class for Boltzmann Machines.

    Args:
        h_range (tuple[float, float], optional): Range for linear weights.
            If ``None``, uses infinite range.
        j_range (tuple[float, float], optional): Range for quadratic weights.
            If ``None``, uses infinite range.
        device (torch.device, optional): Device for tensor construction.
    """

    def __init__(self, h_range=None, j_range=None) -> None:
        super().__init__()
        self.register_buffer(
            "h_range",
            torch.tensor(h_range if h_range is not None else [-torch.inf, torch.inf]),
        )
        self.register_buffer(
            "j_range",
            torch.tensor(j_range if j_range is not None else [-torch.inf, torch.inf]),
        )
        self.register_forward_pre_hook(clip_parameters_hook)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def clip_parameters(self) -> None:
        """Clips linear and quadratic bias weights in-place."""

    def get_ising_matrix(self):
        """Converts the model to Ising format.

        Returns:
            torch.Tensor: Ising matrix.
        """
        self.clip_parameters()
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
        s_negtive: torch.Tensor,
    ) -> torch.Tensor:
        """Objective function whose gradient is equivalent to the gradient of negative log-likelihood.

        Args:
            s_positive (torch.Tensor): Tensor of observed spins (data), shape (b1, N), where b1 is batch size and N is the number of variables.
            s_negtive (torch.Tensor): Tensor of spins sampled from the model, shape (b2, N), where b2 is batch size and N is the number of variables.

        Returns:
            torch.Tensor: Scalar difference between data and model average energy.
        """
        self.clip_parameters()
        return -(self(s_positive).mean() - self(s_negtive).mean())

    def sample(self, sampler) -> torch.Tensor:
        """Samples from the Boltzmann Machine.

        Args:
            sampler: Optimizer used for sampling from the model. The sampler can be kaiwuSDK's CIM or other solvers.

        Returns:
            torch.Tensor: Spins sampled from the model.
        """
        ising_mat = self.get_ising_matrix()
        solution = sampler.solve(ising_mat)
        solution = (solution[:, :-1] + 1) / 2
        solution = torch.FloatTensor(solution)
        solution = solution.to(self.device)
        return solution
