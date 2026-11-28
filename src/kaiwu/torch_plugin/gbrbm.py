# -*- coding: utf-8 -*-
"""
Restricted Gaussian-Bernoulli Boltzmann Machine implementation.

This module implements a restricted Boltzmann machine with one Gaussian
partition and one Bernoulli partition.
"""

import math

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

        # Only parameters the caller did not supply are (re-)initialized, so an
        # explicitly weighted model survives construction (as in the RBM).
        self.init_parameter(
            std=0.01,
            init_var=1,
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

    def init_parameter(
        self,
        init_var: float = 1,
        std: float = 0.01,
        max_fold: float = 16,
        init_quadratic: bool = True,
        init_linear: bool = True,
    ):
        """Initialize model parameters with normal distribution.

        Args:
            init_var (float, optional): Initial variance for Gaussian units.
                Defaults to 1.
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
        init.constant_(self.log_var, np.log(init_var))
        if init_quadratic:
            init.normal_(self.quadratic_coef, 0, std)
            self.quadratic_coef.data.clip_(-bound, bound)
        if init_linear:
            init.normal_(self.linear_bias, 0, std)
            self.linear_bias.data.clip_(-bound, bound)

    def _precision_context(
        self,
        *tensors: torch.Tensor,
    ) -> tuple[torch.dtype, torch.dtype, torch.Tensor]:
        """Return output dtype, working dtype, and clamped ``-0.5 * log_var``."""
        out_dtype = self.log_var.dtype
        for tensor in tensors:
            out_dtype = torch.promote_types(out_dtype, tensor.dtype)
        work_dtype = (
            torch.float32
            if out_dtype in (torch.float16, torch.bfloat16)
            else out_dtype
        )
        min_var = max(float(self.eps), float(torch.finfo(work_dtype).tiny))
        half_neg_lv = -0.5 * self.log_var.to(work_dtype).clamp(
            min=math.log(min_var)
        )
        return out_dtype, work_dtype, half_neg_lv

    @staticmethod
    def _balanced_precision_matmul(
        left: torch.Tensor,
        right: torch.Tensor,
        half_neg_lv: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``(left * exp(2 * half_neg_lv)) @ right`` with balanced scaling."""
        if left.numel() == 0 or right.numel() == 0:
            return torch.zeros(
                (left.shape[0], right.shape[1]),
                dtype=left.dtype,
                device=left.device,
            )
        with torch.no_grad():
            left_max = left.detach().abs().amax(dim=0)
            right_max = right.detach().abs().amax(dim=1)
            h_det = half_neg_lv.detach()
            fmax_log = math.log(torch.finfo(left.dtype).max) - 1.0
            low = torch.minimum(h_det - fmax_log, fmax_log - h_det)
            high = torch.maximum(h_det - fmax_log, fmax_log - h_det)
            both_pos = (left_max > 0) & (right_max > 0)
            safe_left = torch.where(both_pos, left_max, torch.ones_like(left_max))
            safe_right = torch.where(
                both_pos, right_max, torch.ones_like(right_max)
            )
            balanced = 0.5 * (torch.log(safe_left) - torch.log(safe_right))
            raw_delta = torch.where(
                both_pos,
                balanced,
                torch.where(left_max == 0, -h_det, h_det),
            )
            delta = torch.clamp(raw_delta, min=low, max=high)
        eff_h = torch.where(both_pos, half_neg_lv, half_neg_lv.detach())
        scale_left = torch.exp(eff_h - delta)
        scale_right = torch.exp(eff_h + delta)
        return (left * scale_left.unsqueeze(0)) @ (right * scale_right.unsqueeze(1))

    def _centered_whitened_state(
        self,
        s_gaussian: torch.Tensor,
        work_dtype: torch.dtype,
        half_neg_lv: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``(s_gaussian - mu) / std`` with overflow-safe forward and grads."""
        g_work = s_gaussian.to(work_dtype)
        mu_work = self.mu.to(work_dtype)
        inv_std_det = torch.exp(half_neg_lv.detach())
        diff = g_work - mu_work
        finite_diff = torch.isfinite(diff)
        safe_diff = torch.where(finite_diff, diff, torch.zeros_like(diff))
        fb_g = torch.where(finite_diff, torch.zeros_like(g_work), g_work)
        fb_mu = torch.where(finite_diff, torch.zeros_like(g_work), mu_work)
        centered_base = safe_diff * inv_std_det + (
            fb_g * inv_std_det - fb_mu * inv_std_det
        )
        return centered_base * torch.exp(half_neg_lv - half_neg_lv.detach())

    def _masked_precision_interaction(
        self,
        s_gaussian: torch.Tensor,
        weights_bernoulli: torch.Tensor,
        work_dtype: torch.dtype,
        half_neg_lv: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``sum(((s_gaussian / var) @ quadratic_coef) * weights, dim=-1)``."""
        g_work = s_gaussian.to(work_dtype)
        w_work = self.quadratic_coef.to(work_dtype)
        p_work = weights_bernoulli.to(work_dtype)
        active = p_work != 0
        active_rows = active.any(dim=-1, keepdim=True)
        active_cols = active.any(dim=0, keepdim=True)
        g_active = torch.where(active_rows, g_work, torch.zeros_like(g_work))
        w_active = torch.where(active_cols, w_work, torch.zeros_like(w_work))
        raw_logits = self._balanced_precision_matmul(
            g_active, w_active, half_neg_lv
        )
        masked_logits = torch.where(active, raw_logits, torch.zeros_like(p_work))
        return torch.sum(masked_logits * p_work, dim=-1)

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
            out_dtype, work_dtype, half_neg_lv = self._precision_context(s_all)
            s_b = s_bernoulli.to(work_dtype)
            centered = self._centered_whitened_state(
                s_gaussian, work_dtype, half_neg_lv
            )
            interaction_sum = self._masked_precision_interaction(
                s_gaussian, s_b, work_dtype, half_neg_lv
            )
            result = (
                0.5 * torch.sum(centered.square(), dim=-1)
                - interaction_sum
                - s_b @ self.linear_bias.to(work_dtype)
            )
            return result.to(out_dtype)

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
            out_dtype, work_dtype, half_neg_lv = self._precision_context(s_gaussian)
            centered = self._centered_whitened_state(
                s_gaussian, work_dtype, half_neg_lv
            )
            logits = self._balanced_precision_matmul(
                s_gaussian.to(work_dtype),
                self.quadratic_coef.to(work_dtype),
                half_neg_lv,
            ) + self.linear_bias.to(work_dtype)
            result = 0.5 * torch.sum(centered.square(), dim=-1) - torch.sum(
                F.softplus(logits),
                dim=-1,
            )
            return result.to(out_dtype)

    def _to_ising_matrix(self):
        """Convert the Bernoulli part to Ising format.

        This method converts the Bernoulli part of the model to an equivalent
        Ising model representation for optimization purposes.

        Returns:
            numpy.ndarray: Ising matrix representation of the Bernoulli part.
        """
        _, work_dtype, half_neg_lv = self._precision_context()
        mu_work = self.mu.to(work_dtype).unsqueeze(0)
        w_work = self.quadratic_coef.to(work_dtype)
        linear_term = (
            self._balanced_precision_matmul(mu_work, w_work, half_neg_lv).squeeze(0)
            + self.linear_bias.to(work_dtype)
        )
        quadratic_term = self._balanced_precision_matmul(
            w_work.t(), w_work, half_neg_lv
        )
        column_sums = torch.sum(quadratic_term, dim=0)
        num_nodes = self.num_bernoulli
        ising_mat = torch.zeros(
            (num_nodes + 1, num_nodes + 1), device=self.device, dtype=work_dtype
        )
        ising_mat[:-1, :-1] = quadratic_term * 0.125
        ising_linear = linear_term * 0.25 + column_sums * 0.125
        ising_mat[:num_nodes, -1] = ising_linear
        ising_mat[-1, :num_nodes] = ising_linear
        if (
            self.dtype == torch.float16
            and ising_mat.abs().max() <= torch.finfo(torch.float16).max
        ):
            ising_mat = ising_mat.to(torch.float16)
        elif self.dtype not in (work_dtype, torch.float16):
            ising_mat = ising_mat.to(self.dtype)
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
            _, work_dtype, half_neg_lv = self._precision_context(s_gaussian)
            logits = self._balanced_precision_matmul(
                s_gaussian.to(work_dtype),
                self.quadratic_coef.to(work_dtype),
                half_neg_lv,
            ) + self.linear_bias.to(work_dtype)
            prob = torch.sigmoid(logits).to(self.dtype)
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
                s_all[:, : self.num_gaussian] = mu + torch.randn_like(mu) * self.std
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
        with torch.set_grad_enabled(enable_grad):
            out_dtype, work_dtype, half_neg_lv = self._precision_context(s_gaussian)
            bias_work = self.linear_bias.to(work_dtype)
            with torch.no_grad():
                prob_bernoulli = torch.sigmoid(
                    self._balanced_precision_matmul(
                        s_gaussian.detach().to(work_dtype),
                        self.quadratic_coef.detach().to(work_dtype),
                        half_neg_lv.detach(),
                    )
                    + bias_work.detach()
                )
            centered = self._centered_whitened_state(
                s_gaussian, work_dtype, half_neg_lv
            )
            interaction_sum = self._masked_precision_interaction(
                s_gaussian, prob_bernoulli, work_dtype, half_neg_lv
            )
            result = (
                0.5 * torch.sum(centered.square(), dim=-1)
                - interaction_sum
                - prob_bernoulli @ bias_work
            )
            return result.to(out_dtype)
