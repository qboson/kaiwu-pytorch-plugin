# -*- coding: utf-8 -*-
"""
Restricted Gaussian-Bernoulli Boltzmann Machine implementation.

This module implements a restricted Boltzmann machine with one Gaussian
partition and one Bernoulli partition.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

from .abstract_boltzmann_machine import AbstractBoltzmannMachine


class GaussianBernoulliRestrictedBoltzmannMachine(AbstractBoltzmannMachine):
    """Create a Gaussian-Bernoulli Restricted Boltzmann Machine.

    A Gaussian-Bernoulli Restricted Boltzmann Machine has one Gaussian partition
    and one Bernoulli partition. ``is_visible_gaussian`` indicates whether the
    visible nodes correspond to the Gaussian side.

    Args:
        num_visible (int): Number of visible nodes in the model.
        num_hidden (int): Number of hidden nodes in the model.
        is_visible_gaussian (bool, optional): Whether visible nodes are Gaussian
            and hidden nodes are Bernoulli. Defaults to True.
        eps (float, optional): Small value to avoid numerical issues.
            Defaults to 1e-8.
        dtype (torch.dtype, optional): Data type for tensor construction.
            Defaults to torch.float32.
        device (torch.device, optional): Device to construct tensors.
    """

    def __init__(
        self,
        num_visible: int,
        num_hidden: int,
        is_visible_gaussian: bool = True,
        eps=1e-8,
        dtype: torch.dtype = torch.float32,
        linear_bias=None,
        quadratic_coef=None,
        device=None,
    ):
        """Initialize the Restricted Gaussian-Bernoulli Boltzmann Machine."""
        super().__init__(device=device)
        self.dtype = dtype
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.is_visible_gaussian = is_visible_gaussian
        # Keep the existing Gaussian-first internal layout used by the methods below.
        if self.is_visible_gaussian:
            self.num_gaussian = self.num_visible
            self.num_bernoulli = self.num_hidden
        else:
            self.num_gaussian = self.num_hidden
            self.num_bernoulli = self.num_visible
        self.num_nodes = self.num_gaussian + self.num_bernoulli
        self.register_parameter(
            "mu",
            nn.Parameter(
                torch.zeros(
                    self.num_gaussian, dtype=self.dtype, device=self.device
                )
            ),
        )
        self.register_parameter(
            "log_var",
            nn.Parameter(
                torch.ones(self.num_gaussian, dtype=self.dtype, device=self.device)
            ),
        )
        # self.register_parameter(
        #     "U",
        #     nn.Parameter(
        #         torch.zeros(
        #             self.num_gaussian,
        #             self.num_bernoulli,
        #             dtype=self.dtype,
        #             device=self.device,
        #         )
        #     ),
        # )
        # self.register_parameter(
        #     "H",
        #     nn.Parameter(
        #         torch.zeros(
        #             self.num_bernoulli, dtype=self.dtype, device=self.device
        #         )
        #     ),
        # )
        self.quadratic_coef = torch.nn.Parameter(
            quadratic_coef
            if quadratic_coef is not None
            else torch.randn((self.num_gaussian, self.num_bernoulli)).to(self.device) * 0.01
        )
        self.linear_bias = torch.nn.Parameter(
            linear_bias
            if linear_bias is not None
            else torch.zeros(self.num_bernoulli).to(self.device)
        )


        self.init_parameter(std=0.01, init_var=1)

        self.eps = eps

    @property
    def var(self):
        """torch.tensor: Variance of Gaussian units, clipped to avoid numerical issues."""
        return self.log_var.exp().clip(min=self.eps)

    @property
    def std(self):
        """torch.tensor: Standard deviation of Gaussian units."""
        return self.var.sqrt()

    @property
    def diag_precision(self):
        """torch.tensor: Diagonal precision matrix (inverse variance)."""
        return torch.diag(1 / self.var)

    def init_parameter(
        self, init_var: float = 1, std: float = 0.01, max_fold: float = 16
    ):
        """Initialize model parameters with normal distribution.

        Args:
            init_var (float, optional): Initial variance for Gaussian units.
                Defaults to 1.
            std (float, optional): Standard deviation for parameter initialization.
                Defaults to 0.01.
            max_fold (float, optional): Maximum factor for clipping parameters.
                Defaults to 16.
        """
        bound = max_fold * std
        init.normal_(self.mu, 0, std)
        self.get_parameter("mu").data.clip_(-bound, bound)
        init.constant_(self.log_var, np.log(init_var))
        init.normal_(self.quadratic_coef, 0, std)
        self.quadratic_coef.data.clip_(-bound, bound)
        init.normal_(self.linear_bias, 0, std)
        self.linear_bias.data.clip_(-bound, bound)

    def forward(self, s_all: torch.Tensor) -> torch.Tensor:
        """Compute the Hamiltonian.

        Args:
            s_all (torch.tensor): Tensor of shape (B, N), where B is the batch size,
                and N is the number of variables in the model.

        Returns:
            torch.tensor: Hamiltonian of shape (B,).
        """
        return self.energy(s_all, enable_grad=True)

    def energy(self, s_all: torch.Tensor, enable_grad: bool = False) -> torch.Tensor:
        """Compute the Hamiltonian.

        Args:
            s_all (torch.tensor): Tensor of shape (B, N), where B is the batch size,
                and N is the number of variables in the model.
            enable_grad (bool, optional): Whether to enable gradient computation.
                Defaults to False.

        Returns:
            torch.tensor: Hamiltonian of shape (B,).

        """
        with torch.set_grad_enabled(enable_grad):
            s_gaussian = s_all[:, : self.num_gaussian]
            s_bernoulli = s_all[:, self.num_gaussian :]
            return (
                0.5
                * torch.sum((s_gaussian - self.mu).square() / self.var, dim=-1)
                - torch.sum(
                    (s_gaussian / self.var) @ self.quadratic_coef * s_bernoulli, dim=-1
                )
                - s_bernoulli @ self.linear_bias
            )

    def marginal_energy(
        self,
        s_gaussian: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the free Hamiltonian of Gaussian nodes.

        Args:
            s_gaussian (torch.tensor): State of Gaussian nodes.

        Returns:
            torch.tensor: Free Hamiltonian of Gaussian nodes.
        """
        with torch.no_grad():
            return 0.5 * torch.sum(
                (s_gaussian - self.mu).square() / self.var, dim=-1
            ) - torch.sum(
                F.softplus((s_gaussian / self.var) @ self.quadratic_coef + self.linear_bias),
                dim=-1,
            )

    def _to_ising_matrix(self):
        """Convert the Bernoulli part to Ising format.

        This method converts the Bernoulli part of the model to an equivalent
        Ising model representation for optimization purposes.

        Returns:
            numpy.ndarray: Ising matrix representation of the Bernoulli part.
        """
        linear_term = self.quadratic_coef.t() @ (self.mu / self.var) + self.linear_bias
        quadratic_term = self.quadratic_coef.t() @ self.diag_precision @ self.quadratic_coef
        column_sums = torch.sum(quadratic_term, dim=0)
        num_nodes = self.num_bernoulli
        ising_mat = torch.zeros(
            (num_nodes + 1, num_nodes + 1), device=self.device, dtype=self.dtype
        )
        ising_mat[:-1, :-1] = quadratic_term * 0.125
        ising_linear = linear_term * 0.25 + column_sums * 0.125
        ising_mat[:num_nodes, -1] = ising_linear
        ising_mat[-1, :num_nodes] = ising_linear
        return ising_mat.detach().cpu().numpy()

    def infer_from_gaussian(
        self,
        s_gaussian: torch.Tensor,
        binarize: bool = True,
        no_random: bool = False,
    ) -> torch.Tensor:
        """Sample according to given Gaussian nodes.

        Args:
            s_gaussian (torch.tensor): State of Gaussian nodes.
            binarize (bool, optional): If True return 0/1, else return probability.
                Defaults to True.
            no_random (bool, optional): If True return optimal results, else return
                results with randomness. Defaults to False.

        Returns:
            torch.tensor: Complete state tensor with both Gaussian and Bernoulli nodes.

        """
        with torch.no_grad():
            n_sample = s_gaussian.shape[0]
            s_all = torch.zeros(
                n_sample, self.num_nodes, device=self.device, dtype=self.dtype
            )
            s_all[:, : self.num_gaussian] = s_gaussian
            prob = torch.sigmoid((s_gaussian / self.var) @ self.quadratic_coef + self.linear_bias)
            if not binarize:
                s_all[:, self.num_gaussian :] = prob
            elif no_random:
                s_all[:, self.num_gaussian :] = (prob >= 0.5).to(self.dtype)
            else:
                s_all[:, self.num_gaussian :] = (
                    prob > torch.rand_like(prob)
                ).to(self.dtype)
            return s_all

    def infer_from_bernoulli(
        self,
        s_bernoulli: torch.Tensor,
        no_random: bool = False,
    ) -> torch.Tensor:
        """Sample according to given Bernoulli nodes.

        Args:
            s_bernoulli (torch.tensor): State of Bernoulli nodes.
            no_random (bool, optional): If True return optimal results, else return
                results with randomness. Defaults to False.

        Returns:
            torch.tensor: Complete state tensor with both Gaussian and Bernoulli nodes.

        """
        with torch.no_grad():
            n_sample = s_bernoulli.shape[0]
            s_all = torch.zeros(
                n_sample, self.num_nodes, device=self.device, dtype=self.dtype
            )
            s_all[:, self.num_gaussian :] = s_bernoulli
            mu = s_bernoulli @ self.quadratic_coef.t() + self.mu
            if no_random:
                s_all[:, : self.num_gaussian] = mu
            else:
                s_all[:, : self.num_gaussian] = (
                    mu + torch.randn_like(mu) * self.std
                )
            return s_all

    def gibbs_sample(
        self,
        n_step: int,
        n_burnin: int = 0,
        s_gaussian: torch.Tensor | None = None,
        s_bernoulli: torch.Tensor | None = None,
        sampler=None,
        n_sample: int | None = None,
    ) -> torch.Tensor:
        """Perform Gibbs sampling for the Gaussian-Bernoulli RBM.

        Args:
            n_step (int): Number of Gibbs sampling steps to perform.
            n_burnin (int, optional): Number of initial steps to discard.
                Defaults to 0.
            s_gaussian (torch.tensor, optional): Initial Gaussian states.
            s_bernoulli (torch.tensor, optional): Initial Bernoulli states.
            sampler (optional): External sampler used to initialize Bernoulli states.
            n_sample (int, optional): Number of randomly initialized samples.

        Returns:
            torch.tensor: Samples generated through Gibbs sampling.
        """
        n_burnin_ = n_burnin - 1
        with torch.no_grad():
            if s_gaussian is not None:
                n_sample = s_gaussian.shape[0]
                s_all = torch.zeros(
                    n_sample, self.num_nodes, device=self.device, dtype=self.dtype
                )
                s_all[:, : self.num_gaussian] = s_gaussian
                gaussian_start = True
            elif s_bernoulli is not None:
                n_sample = s_bernoulli.shape[0]
                s_all = torch.zeros(
                    n_sample, self.num_nodes, device=self.device, dtype=self.dtype
                )
                s_all[:, self.num_gaussian :] = s_bernoulli
                gaussian_start = False
            elif sampler is not None:
                s_bernoulli = super().sample(sampler).to(
                    device=self.device, dtype=self.dtype
                )
                n_sample = s_bernoulli.shape[0]
                s_all = torch.zeros(
                    n_sample, self.num_nodes, device=self.device, dtype=self.dtype
                )
                s_all[:, self.num_gaussian :] = s_bernoulli
                gaussian_start = False
            else:
                s_gaussian = torch.randn(
                    n_sample, self.num_gaussian, device=self.device, dtype=self.dtype
                )
                s_all = torch.zeros(
                    n_sample, self.num_nodes, device=self.device, dtype=self.dtype
                )
                s_all[:, : self.num_gaussian] = s_gaussian
                gaussian_start = True

            samples = []
            for no_step in range(n_step):
                if gaussian_start:
                    s_gaussian = s_all[:, : self.num_gaussian]
                    s_all = self.infer_from_gaussian(s_gaussian)
                    s_bernoulli = s_all[:, self.num_gaussian :]
                    s_all = self.infer_from_bernoulli(s_bernoulli)
                else:
                    s_bernoulli = s_all[:, self.num_gaussian :]
                    s_all = self.infer_from_bernoulli(s_bernoulli)
                    s_gaussian = s_all[:, : self.num_gaussian]
                    s_all = self.infer_from_gaussian(s_gaussian)
                if no_step >= n_burnin_:
                    samples.append(s_all)
            return torch.concat(samples)

    def sample(self, sampler) -> torch.Tensor:
        """Sample from the model with the abstract sampler interface.

        Args:
            sampler: External optimizer that solves the Bernoulli-side Ising model.

        Returns:
            torch.tensor: Full Gaussian-Bernoulli states.
        """
        s_bernoulli = super().sample(sampler).to(
            device=self.device, dtype=self.dtype
        )
        return self.infer_from_bernoulli(s_bernoulli, no_random=True)

    def positive_phase_energy_expectation(
        self,
        s_gaussian: torch.Tensor,
        enable_grad: bool = True,
    ) -> torch.Tensor:
        """Compute the expected Hamiltonian for observed Gaussian nodes.

        Args:
            s_gaussian (torch.tensor): State of Gaussian nodes.
            enable_grad (bool, optional): Whether to enable gradient computation.
                Defaults to True.

        Returns:
            torch.tensor: Expected Hamiltonian values.
        """
        prob_bernoulli = self.infer_from_gaussian(
            s_gaussian, binarize=False
        )[:, self.num_gaussian :]
        with torch.set_grad_enabled(enable_grad):
            return (
                0.5
                * torch.sum(
                    (s_gaussian - self.mu).square() / self.var, dim=-1
                )
                - torch.sum(
                    (s_gaussian / self.var) @ self.quadratic_coef * prob_bernoulli, dim=-1
                )
                - prob_bernoulli @ self.linear_bias
            )
