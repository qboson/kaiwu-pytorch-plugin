# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Distribution utility classes used in QVAE model"""
import torch
import torch.nn.functional as F


class SmoothingDist:
    """Base class for smoothing distributions.

    Defines the interface for smoothing distributions.
    """

    def pdf(self, zeta):
        """Probability density function r(zeta|z=0).

        Args:
            zeta (torch.Tensor): Input variable.

        Returns:
            torch.Tensor: Probability density value.
        """
        raise NotImplementedError

    def cdf(self, zeta):
        """Cumulative distribution function R(zeta|z=0).

        Args:
            zeta (torch.Tensor): Input variable.

        Returns:
            torch.Tensor: Cumulative distribution value.
        """
        raise NotImplementedError

    def sample(self, shape):
        """Sample from r(zeta|z=0) distribution.

        Args:
            shape (tuple): Sample shape.

        Returns:
            torch.Tensor: Sample result.
        """
        raise NotImplementedError

    def log_pdf(self, zeta):
        """Log probability density log r(zeta|z=0).

        Args:
            zeta (torch.Tensor): Input variable.

        Returns:
            torch.Tensor: Log probability density value.
        """
        raise NotImplementedError


class Exponential(SmoothingDist):
    """Exponential smoothing distribution class.

    Implements PDF, CDF, sampling, and log PDF for exponential smoothing distribution.
    """

    def __init__(self, beta):
        """Initialize exponential distribution.

        Args:
            beta (float or torch.Tensor): Parameter of exponential distribution.
        """
        self.beta = torch.tensor(beta, dtype=torch.float32)

    def pdf(self, zeta: torch.Tensor) -> torch.Tensor:
        """Probability density function.

        Args:
            zeta (torch.Tensor): Input variable.

        Returns:
            torch.Tensor: Probability density value.
        """
        return self.beta * torch.exp(-self.beta * zeta) / (1 - torch.exp(-self.beta))

    def cdf(self, zeta: torch.Tensor) -> torch.Tensor:
        """Cumulative distribution function.

        Args:
            zeta (torch.Tensor): Input variable.

        Returns:
            torch.Tensor: Cumulative distribution value.
        """
        return (1.0 - torch.exp(-self.beta * zeta)) / (1 - torch.exp(-self.beta))

    def sample(self, shape: tuple) -> torch.Tensor:
        """Sampling.

        Args:
            shape (tuple): Sample shape.

        Returns:
            torch.Tensor: Sample result.
        """
        rho = torch.rand(shape)
        zeta = -torch.log(1.0 - (1.0 - torch.exp(-self.beta)) * rho) / self.beta
        return zeta

    def log_pdf(self, zeta: torch.Tensor) -> torch.Tensor:
        """Log probability density.

        Args:
            zeta (torch.Tensor): Input variable.

        Returns:
            torch.Tensor: Log probability density value.
        """
        return (
            torch.log(self.beta)
            - self.beta * zeta
            - torch.log(1 - torch.exp(-self.beta))
        )


class DistUtil:
    """Base class for distribution utilities."""

    def reparameterize(self, is_training):
        """Reparameterization sampling.

        Args:
            is_training (bool): Whether in training mode.

        Returns:
            torch.Tensor: Sample result.
        """
        raise NotImplementedError

    def entropy(self):
        """Entropy calculation.

        Returns:
            torch.Tensor: Entropy value.
        """
        raise NotImplementedError


def sigmoid_cross_entropy_with_logits(logits, labels):
    """Compute sigmoid cross-entropy loss.

    Args:
        logits (torch.Tensor): Logits.
        labels (torch.Tensor): Labels.

    Returns:
        torch.Tensor: Sigmoid cross-entropy loss.
    """
    return logits - logits * labels + F.softplus(-logits)


class FactorialBernoulliUtil(DistUtil):
    """Factorial Bernoulli distribution utility class.

    Used for handling probability distributions of binary random variables.
    """

    def __init__(self, param):
        """Initialize factorial Bernoulli distribution.

        Args:
            param (torch.Tensor): Distribution parameter.
        """
        super().__init__()
        self.logit_mu = param

    def reparameterize(self, is_training: bool) -> torch.Tensor:
        """Sample from Bernoulli distribution.

        Only used in test mode, as reparameterization of Bernoulli distribution
        is not differentiable during training.

        Args:
            is_training (bool): Whether in training mode.

        Returns:
            torch.Tensor: Sample result.

        Raises:
            NotImplementedError: Raised when is_training is True.
        """
        if is_training:
            raise NotImplementedError(
                "Reparameterization of Bernoulli distribution is not differentiable during training"
            )
        device = self.logit_mu.device
        q = torch.sigmoid(self.logit_mu)
        rho = torch.rand_like(q, device=device)
        z = (rho < q).float()
        return z

    def entropy(self):
        """Compute entropy of Bernoulli distribution.

        Returns:
            torch.Tensor: Entropy value.
        """
        mu = torch.sigmoid(self.logit_mu)
        ent = sigmoid_cross_entropy_with_logits(logits=self.logit_mu, labels=mu)
        return ent

    def log_prob_per_var(self, samples):
        """Compute log probability of samples under the distribution.

        Args:
            samples (torch.Tensor): Sample matrix, shape (num_samples, num_vars).

        Returns:
            torch.Tensor: Log probability matrix, shape (num_samples, num_vars).
        """
        log_prob = -sigmoid_cross_entropy_with_logits(
            logits=self.logit_mu, labels=samples
        )
        return log_prob


class MixtureGeneric(FactorialBernoulliUtil):
    """Mixture distribution class.

    Creates a mixture of two overlapping distributions by setting the logits of
    the factorial Bernoulli distribution defined on the z component.
    Can work with any smoothing distribution inheriting from SmoothingDist.
    """

    num_param = 1

    def __init__(self, param, smoothing_dist_beta):
        """Initialize mixture distribution.

        Args:
            param (torch.Tensor): Distribution parameter.
            smoothing_dist_beta (float): Beta parameter of smoothing distribution.
        """
        super().__init__(param)
        self.smoothing_dist = Exponential(smoothing_dist_beta)

    def reparameterize(self, is_training: bool) -> torch.Tensor:
        """Sample from the mixture of two overlapping distributions using ancestral sampling.

        Uses the implicit gradient idea to compute the gradient of samples with respect to logit_q.
        This idea is proposed in DVAE# sec 3.4.
        This function does not implement gradients of samples with respect to beta or other
        parameters of the smoothing transformation.

        Args:
            is_training (bool): Whether in training mode.

        Returns:
            torch.Tensor: Sample result.
        """
        q = torch.sigmoid(self.logit_mu)

        # Sample from Bernoulli distribution
        z = super().reparameterize(is_training=False)
        shape = z.shape

        # Sample from smoothing distribution
        zeta = self.smoothing_dist.sample(shape)
        zeta = zeta.to(z.device)

        zeta = torch.where(z == 0.0, zeta, 1.0 - zeta)

        # Compute PDF and CDF
        pdf_0 = self.smoothing_dist.pdf(zeta)
        pdf_1 = self.smoothing_dist.pdf(1.0 - zeta)
        cdf_0 = self.smoothing_dist.cdf(zeta)
        cdf_1 = 1.0 - self.smoothing_dist.cdf(1.0 - zeta)

        # Compute gradient
        grad_q = (cdf_0 - cdf_1) / (q * pdf_1 + (1 - q) * pdf_0)
        grad_q = grad_q.detach()
        grad_term = grad_q * q
        grad_term = grad_term - grad_term.detach()
        # Only let the gradient flow to q, not zeta itself
        zeta = zeta.detach() + grad_term

        return zeta

    def log_prob_per_var(self, samples: torch.Tensor) -> torch.Tensor:
        """Compute log probability of samples under the mixture of overlapping distributions.

        Args:
            samples (torch.Tensor): Sample matrix, shape (num_samples, num_vars).

        Returns:
            torch.Tensor: Log probability matrix, shape (num_samples, num_vars).
        """
        q = torch.sigmoid(self.logit_mu)
        pdf_0 = self.smoothing_dist.pdf(samples)
        pdf_1 = self.smoothing_dist.pdf(1.0 - samples)
        log_prob = torch.log(q * pdf_1 + (1 - q) * pdf_0)
        return log_prob

    def log_ratio(self, zeta: torch.Tensor) -> torch.Tensor:
        """Compute log_ratio required for KL gradient (proposed in DVAE++).

        Args:
            zeta (torch.Tensor): Approximate posterior samples.

        Returns:
            torch.Tensor: log r(zeta|z=1) - log r(zeta|z=0).
        """
        log_pdf_0 = self.smoothing_dist.log_pdf(zeta)
        log_pdf_1 = self.smoothing_dist.log_pdf(1.0 - zeta)
        log_ratio = log_pdf_1 - log_pdf_0
        return log_ratio
