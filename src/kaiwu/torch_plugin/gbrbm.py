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
        covariance (array_like, optional): Positive-definite covariance matrix
            of shape ``(num_gaussian, num_gaussian)`` describing the
            correlations among the Gaussian units. When given, the Gaussian
            units are modeled with the full covariance
            ``cov_chol @ diag(var) @ cov_chol.T`` where ``cov_chol`` is a
            fixed buffer derived from the supplied matrix and the diagonal
            ``var`` (from ``log_var``) stays trainable. Defaults to None,
            which keeps the historical independent (diagonal) Gaussian units.
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
        covariance=None,
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
                torch.zeros(self.num_gaussian, dtype=self.dtype, device=self.device)
            ),
        )
        self.register_parameter(
            "log_var",
            nn.Parameter(
                torch.ones(self.num_gaussian, dtype=self.dtype, device=self.device)
            ),
        )
        self.quadratic_coef = torch.nn.Parameter(
            (
                quadratic_coef
                if quadratic_coef is not None
                else torch.randn((self.num_gaussian, self.num_bernoulli))
            ).to(dtype=self.dtype, device=self.device)
        )
        self.linear_bias = torch.nn.Parameter(
            (
                linear_bias
                if linear_bias is not None
                else torch.zeros(self.num_bernoulli)
            ).to(dtype=self.dtype, device=self.device)
        )

        # Full covariance support: the Gaussian covariance is factorized as
        # cov_chol @ diag(var) @ cov_chol.T with a fixed cov_chol derived
        # from the supplied matrix, so the diagonal var stays trainable.
        cov_chol = None
        cov_chol_inv = None
        init_var = 1
        if covariance is not None:
            cov_t = torch.as_tensor(covariance, dtype=self.dtype, device=self.device)
            if cov_t.shape != (self.num_gaussian, self.num_gaussian):
                raise ValueError(
                    "covariance must have shape "
                    f"({self.num_gaussian}, {self.num_gaussian}), "
                    f"got {tuple(cov_t.shape)}"
                )
            if not torch.allclose(cov_t, cov_t.t(), rtol=1e-5, atol=1e-8):
                raise ValueError("covariance must be symmetric")
            try:
                base_chol = torch.linalg.cholesky(cov_t)
            except RuntimeError as err:
                raise ValueError(
                    "covariance must be positive definite"
                ) from err
            var0 = torch.diagonal(cov_t)
            # cov_chol = chol(covariance) / sqrt(diag(covariance)) so that
            # cov_chol @ diag(diag(covariance)) @ cov_chol.T equals the
            # supplied covariance and diagonal inputs reduce to the identity.
            cov_chol = base_chol / var0.sqrt().unsqueeze(0)
            cov_chol_inv = torch.linalg.inv(cov_chol)
            init_var = var0
        self.register_buffer("cov_chol", cov_chol)
        self.register_buffer("cov_chol_inv", cov_chol_inv)

        # Only parameters the caller did not supply are (re-)initialized, so an
        # explicitly weighted model survives construction (as in the RBM).
        self.init_parameter(
            std=0.01,
            init_var=init_var,
            init_quadratic=quadratic_coef is None,
            init_linear=linear_bias is None,
        )

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

    def _precision_scaled(self, s_gaussian: torch.Tensor) -> torch.Tensor:
        """Map Gaussian states through the precision of the Gaussian units.

        With diagonal covariance this returns ``s_gaussian / var``. With a
        full covariance ``cov_chol`` it returns
        ``s_gaussian @ inv(covariance)`` computed as
        ``(s_gaussian @ cov_chol_inv.T / var) @ cov_chol_inv``.
        """
        if self.cov_chol_inv is None:
            return s_gaussian / self.var
        return (
            (s_gaussian @ self.cov_chol_inv.t()) / self.var
        ) @ self.cov_chol_inv

    def _gaussian_prior_energy(self, s_gaussian: torch.Tensor) -> torch.Tensor:
        """Quadratic-form energy of the Gaussian prior.

        With diagonal covariance this returns
        ``0.5 * sum((s_gaussian - mu)^2 / var)``. With a full covariance it
        returns ``0.5 * sum(((s_gaussian - mu) @ cov_chol_inv.T / std)^2)``.
        """
        centered = s_gaussian - self.mu
        if self.cov_chol_inv is None:
            return 0.5 * torch.sum(centered.square() / self.var, dim=-1)
        white = (centered @ self.cov_chol_inv.t()) / self.std
        return 0.5 * white.square().sum(dim=-1)

    def _std_correlated_noise(self, eps: torch.Tensor) -> torch.Tensor:
        """Correlate standard-normal noise with the Gaussian covariance.

        With diagonal covariance this returns ``eps * std``. With a full
        covariance it returns ``(eps * std) @ cov_chol.T``, whose covariance
        is ``cov_chol @ diag(var) @ cov_chol.T``.
        """
        if self.cov_chol is None:
            return eps * self.std
        return (eps * self.std) @ self.cov_chol.t()

    def init_parameter(
        self,
        init_var=1,
        std: float = 0.01,
        max_fold: float = 16,
        init_quadratic: bool = True,
        init_linear: bool = True,
    ):
        """Initialize model parameters with normal distribution.

        Args:
            init_var (float or torch.Tensor, optional): Initial variance for
                the Gaussian units, either a scalar or a per-unit tensor of
                shape ``(num_gaussian,)``. Defaults to 1.
            std (float, optional): Standard deviation for parameter initialization.
                Defaults to 0.01.
            max_fold (float, optional): Maximum factor for clipping parameters.
                Defaults to 16.
            init_quadratic (bool, optional): Whether to (re-)initialize the
                quadratic coefficients. Defaults to True.
            init_linear (bool, optional): Whether to (re-)initialize the linear
                bias. Defaults to True.
        """
        bound = max_fold * std
        init.normal_(self.mu, 0, std)
        self.get_parameter("mu").data.clip_(-bound, bound)
        if isinstance(init_var, torch.Tensor):
            if init_var.shape != self.log_var.shape:
                raise ValueError(
                    "init_var must be a scalar or a tensor of shape "
                    f"{tuple(self.log_var.shape)}"
                )
            self.log_var.data.copy_(init_var.log())
        else:
            init.constant_(self.log_var, np.log(init_var))
        if init_quadratic:
            init.normal_(self.quadratic_coef, 0, std)
            self.quadratic_coef.data.clip_(-bound, bound)
        if init_linear:
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
                self._gaussian_prior_energy(s_gaussian)
                - torch.sum(
                    self._precision_scaled(s_gaussian)
                    @ self.quadratic_coef
                    * s_bernoulli,
                    dim=-1,
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
            return self._gaussian_prior_energy(s_gaussian) - torch.sum(
                F.softplus(
                    self._precision_scaled(s_gaussian) @ self.quadratic_coef
                    + self.linear_bias
                ),
                dim=-1,
            )

    def _to_ising_matrix(self):
        """Convert the Bernoulli part to Ising format.

        This method converts the Bernoulli part of the model to an equivalent
        Ising model representation for optimization purposes.

        Returns:
            numpy.ndarray: Ising matrix representation of the Bernoulli part.
        """
        linear_term = (
            self.quadratic_coef.t() @ self._precision_scaled(self.mu)
            + self.linear_bias
        )
        quadratic_term = self.quadratic_coef.t() @ self._precision_scaled(
            self.quadratic_coef.t()
        ).t()
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
            prob = torch.sigmoid(
                self._precision_scaled(s_gaussian) @ self.quadratic_coef
                + self.linear_bias
            )
            if not binarize:
                s_all[:, self.num_gaussian :] = prob
            elif no_random:
                s_all[:, self.num_gaussian :] = (prob >= 0.5).to(self.dtype)
            else:
                s_all[:, self.num_gaussian :] = (prob > torch.rand_like(prob)).to(
                    self.dtype
                )
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
                s_all[:, : self.num_gaussian] = mu + self._std_correlated_noise(
                    torch.randn_like(mu)
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
                s_bernoulli = (
                    super().sample(sampler).to(device=self.device, dtype=self.dtype)
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
        s_bernoulli = super().sample(sampler).to(device=self.device, dtype=self.dtype)
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
        prob_bernoulli = self.infer_from_gaussian(s_gaussian, binarize=False)[
            :, self.num_gaussian :
        ]
        with torch.set_grad_enabled(enable_grad):
            return (
                self._gaussian_prior_energy(s_gaussian)
                - torch.sum(
                    self._precision_scaled(s_gaussian)
                    @ self.quadratic_coef
                    * prob_bernoulli,
                    dim=-1,
                )
                - prob_bernoulli @ self.linear_bias
            )
