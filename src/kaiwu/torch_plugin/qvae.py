# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""
Quantum Variational Autoencoder (QVAE) model.

This module contains the QVAE class and supporting utilities for training
a QVAE with a Boltzmann machine prior.
"""

import abc
import logging

import torch
from torch import nn
import torch.nn.functional as F

from .qvae_dist_util import MixtureGeneric, FactorialBernoulliUtil

logger = logging.getLogger(__name__)


# Base Class for all AutoEncoder models
class AutoEncoderBase(nn.Module):
    """
    Base class for AutoEncoders, providing common initialization and interfaces.

    Args:
        input_dimension (int or list of int): Dimensionality of input features.
            If list, only first element is used for now.
        activation_fct (callable, optional): Activation function for hidden layers.
        config (object): Configuration object with hyperparameters.
            Must contain `num_latent_units` (int > 0) and `loss_type` (str).
        ``**kwargs``: Additional keyword arguments for nn.Module.
    """

    def __init__(
        self, input_dimension=None, activation_fct=None, config=None, **kwargs
    ):
        super().__init__(**kwargs)

        # Validate and normalize input dimension
        if isinstance(input_dimension, list):
            assert (
                len(input_dimension) > 0
            ), "Input dimension not defined, needed for model structure"
        else:
            assert (
                input_dimension > 0
            ), "Input dimension not defined, needed for model structure"
            input_dimension = [input_dimension]  # wrap in list for consistent handling

        assert config is not None, "Config not defined"
        assert (
            config.num_latent_units is not None and config.num_latent_units > 0
        ), "Latent dimension must be >0"
        assert hasattr(
            config, "loss_type"
        ), "Config must contain loss_type (e.g., 'bernoulli' or 'mse')"

        self._model_type = None
        self.config = config
        self._latent_dimensions = config.num_latent_units
        self._input_dimension = input_dimension[0]  # single input dimension
        self._activation_fct = activation_fct
        self._dataset_mean = None  # for Bernoulli bias correction

    @abc.abstractmethod
    def _create_encoder(self):
        """Create encoder network. Must be implemented in subclasses."""
        raise NotImplementedError

    @abc.abstractmethod
    def _create_decoder(self):
        """Create decoder network. Must be implemented in subclasses."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, x):
        """Forward pass. Must be implemented in subclasses."""
        raise NotImplementedError

    def set_dataset_mean(self, mean):
        """
        Set dataset mean for bias correction.

        Args:
            mean (torch.Tensor): Mean of the training data (shape: input_dim).
        """
        self._dataset_mean = mean

    def __repr__(self):
        parameter_string = "\n".join([str(par) for par in self.__dict__.items()])
        return parameter_string


class QVAE(AutoEncoderBase):
    """
    Quantum Variational Autoencoder integrated into AutoEncoderBase framework.

    Args:
        input_dimension (int or list of int): Dimensionality of input features.
        activation_fct (callable, optional): Activation function for hidden layers.
        config (object): Configuration with ``num_latent_units`` (int),
            ``loss_type`` (``'bernoulli'`` or ``'mse'``), ``dist_beta``
            (float), ``kl_beta`` (float), and ``weight_decay`` (float).
        encoder (nn.Module, optional): Pre-created encoder. Created by the
            subclass factory when omitted.
        decoder (nn.Module, optional): Pre-created decoder. Created by the
            subclass factory when omitted.
        bm (nn.Module, optional): Pre-created Boltzmann machine. Created by the
            subclass factory when omitted.
        sampler (object, optional): Pre-created sampler. Created by the subclass
            factory when omitted.
        sampler_type (str): Type of sampler for BM ('sa' or 'cim').
    """

    def __init__(
        self,
        input_dimension,
        activation_fct,
        config,
        encoder=None,
        decoder=None,
        bm=None,
        sampler=None,
        sampler_type=None,
    ):
        super().__init__(input_dimension, activation_fct, config)
        self._model_type = "QVAE"  # for identification, can be used in ModelTuner
        self.sampler_type = sampler_type or getattr(self.config, "sampler_type", "sa")

        # Parameters from config
        self.dist_beta = self.config.dist_beta
        self.kl_beta = self.config.kl_beta
        self.weight_decay = self.config.weight_decay

        # Explicitly supplied components take precedence over subclass factories.
        self.encoder = encoder
        self.decoder = decoder
        self.bm = bm
        self.sampler = sampler

        # Bernoulli reconstruction bias is model state, but not a trainable parameter.
        self.register_buffer("_train_bias", torch.zeros(self._input_dimension))
        self.last_recon_loss = None
        self.last_kl_loss = None
        self.create_networks()

    def create_networks(self):
        """Create components that were not supplied to the constructor."""
        if self.encoder is None:
            self.encoder = self._create_encoder()
        if self.decoder is None:
            self.decoder = self._create_decoder()
        if self.bm is None:
            self.bm = self._create_bm()
        if self.sampler is None:
            self.sampler = self._create_sampler(self.sampler_type)
        if self._dataset_mean is not None:
            self.set_train_bias(self._dataset_mean)

    # -------- Abstract methods for subclasses --------
    @abc.abstractmethod
    def _create_encoder(self):
        pass

    @abc.abstractmethod
    def _create_decoder(self):
        pass

    @abc.abstractmethod
    def _create_bm(self):
        pass

    def energy(self, x, loss_type=None):
        """Compute the Boltzmann-machine energy for each input sample.

        Args:
            x (torch.Tensor): Input samples with the input feature dimension.
            loss_type (str, optional): Loss type to validate against the model
                configuration. Defaults to the configured loss type.

        Returns:
            torch.Tensor: One energy value per input sample.
        """
        configured_loss_type = self.config.loss_type
        if loss_type is not None and loss_type != configured_loss_type:
            raise ValueError("loss_type must match model.config.loss_type")

        x = x.view(-1, self._input_dimension)
        if configured_loss_type == "bernoulli" and self._dataset_mean is not None:
            x = x - torch.as_tensor(self._dataset_mean, dtype=x.dtype, device=x.device)
        elif configured_loss_type != "mse":
            raise ValueError(f"Unsupported loss type: {configured_loss_type}")

        q = self.encoder(x)
        return self.bm((q > 0).float())

    @abc.abstractmethod
    def _create_sampler(self, sampler_type):
        pass

    # -------- Public methods --------
    def set_train_bias(self, mean):
        """Compute train bias from dataset mean for Bernoulli reconstruction."""
        mean = torch.as_tensor(
            mean,
            dtype=self._train_bias.dtype,
            device=self._train_bias.device,
        )
        if mean.numel() == 1:
            mean = mean.expand(self._input_dimension)
        elif mean.numel() != self._input_dimension:
            raise ValueError(
                "Dataset mean must be a scalar or match the input dimension."
            )
        clipped_mean = torch.clamp(mean.reshape(-1), 0.001, 0.999).detach()
        self._train_bias.copy_(-torch.log(1 / clipped_mean - 1))

    def forward(self, x):
        """
        Forward pass through the QVAE.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            tuple: Reconstructed logits, posterior distribution, encoder logits,
                and reparameterized latent sample.
        """
        x = x.view(-1, self._input_dimension)

        # For Bernoulli data, we optionally subtract dataset mean
        if self.config.loss_type == "bernoulli":
            encoder_x = x
            if self._dataset_mean is not None:
                encoder_x = encoder_x - torch.as_tensor(
                    self._dataset_mean,
                    dtype=x.dtype,
                    device=x.device,
                )
        elif self.config.loss_type == "mse":
            encoder_x = x
        else:
            raise ValueError(f"Unsupported loss type: {self.config.loss_type}")

        q = self.encoder(encoder_x)  # encoder must implement forward
        posterior, zeta = self.posterior(q, self.config.dist_beta)

        recon_x = self.decoder(zeta)  # decoder must implement forward

        # Add Bernoulli bias if needed
        if self.config.loss_type == "bernoulli":
            recon_x = recon_x + self._train_bias

        return recon_x, posterior, q, zeta

    def loss(self, x, recon_x, posterior):
        """Compute total loss (reconstruction + KL + weight decay).

        Args:
            x (torch.Tensor): Input tensor (batch_size, input_dim).
            recon_x (torch.Tensor): Reconstructed logits (batch_size, input_dim).
            posterior (MixtureGeneric): Posterior distribution object.
            q (torch.Tensor): Encoder logits (batch_size, latent_dim).
            zeta (torch.Tensor): Latent sample (batch_size, latent_dim).

        Returns:
            torch.Tensor: Total loss scalar.

        Raises:
            ValueError: If loss_type is not supported.
        """
        if self.config.loss_type == "mse":
            recon_loss = F.mse_loss(
                recon_x, x.view(-1, self._input_dimension), reduction="sum"
            ) / x.size(0)
        elif self.config.loss_type == "bernoulli":  # bernoulli
            # recon_loss = F.binary_cross_entropy_with_logits(
            #     recon_x,
            #     x.view(-1, self._input_dimension),
            #     reduction='sum'
            # ) / x.size(0)
            output_dist = FactorialBernoulliUtil(recon_x)
            recon_loss = -output_dist.log_prob_per_var(x).sum(dim=1).mean()
        else:
            raise ValueError(f"Unsupported loss type: {self.config.loss_type}")

        # KL divergence
        kl_loss = self._kl_dist_from(posterior).mean()

        # Weight decay
        wd_loss = self._weight_decay_loss()

        # Total loss
        total_loss = recon_loss + self.kl_beta * kl_loss + wd_loss
        return total_loss

    def bm_loss(self, q, bm_weight_decay=0.0):
        """Compute BM loss for updating BM parameters separately.

        Args:
            q (torch.Tensor): Encoder output logits (batch_size, latent_dim).
                Must be detached to prevent gradients flowing to encoder.

            bm_weight_decay (float, optional): L2 regularization coefficient for BM parameters.

        Returns:
            torch.Tensor: BM loss scalar.
        """
        # Use hard binary samples from q (threshold 0) or sampling from sigmoid(q)
        # positive_state = (q.detach() > 0).float()
        # loss = self.bm.objective(positive_state, self.bm.sample(self.sampler))
        # Alternatively, use the probabilities from sigmoid(q) for a softer loss signal to BM
        loss = self.bm.objective(
            torch.sigmoid(q.detach()), self.bm.sample(self.sampler)
        )
        if bm_weight_decay > 0:
            if hasattr(self.bm, "quadratic_coef"):
                loss += bm_weight_decay * torch.sum(self.bm.quadratic_coef**2)
            if hasattr(self.bm, "linear_bias"):
                loss += bm_weight_decay * 0.5 * torch.sum(self.bm.linear_bias**2)
        return loss

    def posterior(self, q_logits, beta):
        """Compute posterior distribution and reparameterized sample.

        Args:
            q_logits (torch.Tensor): Encoder output logits (batch_size, latent_dim).
            beta (float): Mixture parameter for MixtureGeneric.

        Returns:
            tuple: Posterior distribution and reparameterized latent sample.
        """
        posterior_dist = MixtureGeneric(q_logits, beta)
        zeta = posterior_dist.reparameterize(self.training)
        return posterior_dist, zeta

    def _kl_dist_from(self, posterior):
        """Compute KL divergence: cross_entropy - entropy.

        Args:
            posterior (MixtureGeneric): Posterior distribution.

        Returns:
            torch.Tensor: KL divergence per sample (batch_size,).
        """
        entropy = torch.sum(posterior.entropy(), dim=1)
        logit_q = posterior.logit_mu
        cross_entropy = self._cross_entropy(logit_q)
        return cross_entropy - entropy

    def _cross_entropy(self, logit_q):
        """Compute cross-entropy term for KL divergence.

        Args:
            logit_q (torch.Tensor): Logits from encoder (batch_size, latent_dim).

        Returns:
            torch.Tensor: Cross-entropy per sample (batch_size,).
        """
        q_prob = torch.sigmoid(logit_q)
        positive = self.bm(q_prob).mean()
        neg_samples = self.bm.sample(self.sampler)
        negative = self.bm(neg_samples).mean()
        return positive - negative

    def _weight_decay_loss(self):
        """Compute L2 regularization on BM parameters.

        Returns:
            torch.Tensor: Weight decay loss scalar.
        """
        wd = 0.0
        if hasattr(self.bm, "quadratic_coef"):
            wd += self.weight_decay * torch.sum(self.bm.quadratic_coef**2)
        if hasattr(self.bm, "linear_bias"):
            wd += self.weight_decay * 0.5 * torch.sum(self.bm.linear_bias**2)
        return wd


class Q_SVI:
    """
    Quantum Stochastic Variational Inference (Q-SVI) training engine.

    Aligns with the pyro.infer.SVI interface (model, guide, loss, step(x))
    while maintaining the two-stage alternating optimization sequence for
    Quantum Variational Autoencoders (QVAE).

    Args:
        model (AutoEncoderBase or nn.Module): The QVAE model instance.
        guide (callable, optional): Variational guide. If None, uses the model's
            internal encoder and posterior distribution.
        optim (torch.optim.Optimizer or tuple or dict, optional): Primary optimizer
            for the encoder/decoder, or a container of (vae_optim, bm_optim).
        loss (callable, optional): Objective loss function taking (model, guide, x)
            or (x, recon_x, posterior). Defaults to model.loss.
        bm_optim (torch.optim.Optimizer, optional): Optimizer for the Boltzmann
            Machine parameters. If provided, two-stage alternating optimization
            is enabled.
        bm_weight_decay (float, optional): Weight decay coefficient for BM loss.
            Defaults to 0.0.
    """

    def __init__(
        self,
        model,
        guide=None,
        optim=None,
        loss=None,
        bm_optim=None,
        bm_weight_decay=0.0,
    ):
        self.model = model
        self.guide = guide
        self.loss_fn = loss
        self.bm_weight_decay = float(bm_weight_decay)

        self.vae_optim = None
        self.bm_optim = None
        self.use_two_optimisers = False

        self._configure_optimizers(optim, bm_optim)

        self.last_loss = 0.0
        self.last_vae_loss = 0.0
        self.last_bm_loss = 0.0

    def _configure_optimizers(self, optim, bm_optim=None):
        """Configure VAE and BM optimizers from flexible input types."""
        if isinstance(optim, (tuple, list)):
            self.vae_optim = optim[0]
            self.bm_optim = optim[1] if len(optim) > 1 else bm_optim
        elif isinstance(optim, dict):
            self.vae_optim = optim.get("vae") or optim.get("optim")
            self.bm_optim = optim.get("bm") or bm_optim
        else:
            self.vae_optim = optim
            self.bm_optim = bm_optim

        self.use_two_optimisers = self.bm_optim is not None

    def set_optimizers(self, vae_optim=None, bm_optim=None):
        """Update or register optimizers on the SVI engine.

        Args:
            vae_optim (torch.optim.Optimizer, optional): VAE optimizer.
            bm_optim (torch.optim.Optimizer, optional): BM optimizer.
        """
        if vae_optim is not None:
            self.vae_optim = vae_optim
        if bm_optim is not None:
            self.bm_optim = bm_optim
        self.use_two_optimisers = self.bm_optim is not None

    def _compute_forward_and_loss(self, x, *args, **kwargs):
        """Internal helper to compute forward pass and loss consistently."""
        if self.guide is not None:
            posterior, q, zeta = self.guide(x, *args, **kwargs)
            recon_x = self.model.decoder(zeta)
            if getattr(self.model, "config", None) and getattr(self.model.config, "loss_type", None) == "bernoulli":
                if hasattr(self.model, "_train_bias"):
                    recon_x = recon_x + self.model._train_bias
        else:
            output = self.model(x, *args, **kwargs)
            recon_x, posterior, q, _ = output

        if self.loss_fn is not None:
            try:
                vae_loss = self.loss_fn(x, recon_x, posterior)
            except TypeError:
                vae_loss = self.loss_fn(self.model, self.guide, x, *args, **kwargs)
        else:
            vae_loss = self.model.loss(x, recon_x, posterior)

        return vae_loss, q

    def step(self, x, *args, **kwargs):
        """Take a single optimization step on batch x.

        Maintains the two-stage alternating optimization sequence:
        1. Zero grad on VAE optimizer, forward pass through model, compute loss,
           backward pass, step VAE optimizer.
        2. If BM optimizer is present, zero grad on BM optimizer, compute
           bm_loss on q.detach(), backward pass, step BM optimizer.

        Args:
            x (torch.Tensor): Input batch tensor.
            *args: Additional positional arguments for model forward.
            **kwargs: Additional keyword arguments for model forward.

        Returns:
            float: Total scalar step loss (VAE loss + BM loss if two-stage).
        """
        if self.vae_optim is not None:
            self.vae_optim.zero_grad()

        vae_loss, q = self._compute_forward_and_loss(x, *args, **kwargs)
        vae_loss.backward()

        if self.vae_optim is not None:
            self.vae_optim.step()

        self.last_vae_loss = vae_loss.item()
        total_loss = self.last_vae_loss

        if self.use_two_optimisers and hasattr(self.model, "bm_loss"):
            self.bm_optim.zero_grad()
            bm_loss = self.model.bm_loss(q.detach(), self.bm_weight_decay)
            bm_loss.backward()
            self.bm_optim.step()

            self.last_bm_loss = bm_loss.item()
            total_loss += self.last_bm_loss
        else:
            self.last_bm_loss = 0.0

        self.last_loss = total_loss
        return float(total_loss)

    def evaluate_loss(self, x, *args, **kwargs):
        """Evaluate loss on batch x without parameter updates.

        Args:
            x (torch.Tensor): Input batch tensor.
            *args: Additional positional arguments for model forward.
            **kwargs: Additional keyword arguments for model forward.

        Returns:
            float: Total evaluated scalar loss.
        """
        with torch.no_grad():
            vae_loss, q = self._compute_forward_and_loss(x, *args, **kwargs)
            total_loss = vae_loss.item()
            if self.use_two_optimisers and hasattr(self.model, "bm_loss"):
                bm_loss = self.model.bm_loss(q.detach(), self.bm_weight_decay)
                total_loss += bm_loss.item()
            return float(total_loss)

    evaluate = evaluate_loss

    def get_last_loss_dict(self):
        """Return the dictionary of loss components from the last step."""
        return {
            "total_loss": self.last_loss,
            "vae_loss": self.last_vae_loss,
            "bm_loss": self.last_bm_loss,
        }


# Public alias for naming consistency
QSVI = Q_SVI

