# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import abc
import logging

from .qvae_dist_util import MixtureGeneric, FactorialBernoulliUtil

logger = logging.getLogger(__name__)
torch.manual_seed(42)

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
        **kwargs: Additional keyword arguments for nn.Module.    
    """
    def __init__(self, input_dimension=None, activation_fct=None, config=None, **kwargs):
        super(AutoEncoderBase,self).__init__(**kwargs)
        
        # Validate and normalize input dimension
        if isinstance(input_dimension, list):
            assert len(input_dimension) > 0, "Input dimension not defined, needed for model structure"
        else:
            assert input_dimension > 0, "Input dimension not defined, needed for model structure"
            input_dimension = [input_dimension]  # wrap in list for consistent handling
            
        assert config is not None, "Config not defined"
        assert config.num_latent_units is not None and config.num_latent_units > 0, "Latent dimension must be >0"
        assert hasattr(config, "loss_type"), "Config must contain loss_type (e.g., 'bernoulli' or 'mse')"        

        self._model_type = None
        self._config = config
        self._latent_dimensions = config.num_latent_units
        self._input_dimension = input_dimension[0]  # single input dimension
        self._activation_fct = activation_fct
        self._dataset_mean = None   # for Bernoulli bias correction

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
    
    def set_dataset_mean(self,mean):
        """
        Set dataset mean for bias correction.
        
        Args:
            mean (torch.Tensor): Mean of the training data (shape: input_dim).
        """
        self._dataset_mean=mean    

    def __repr__(self):
        parameter_string="\n".join([str(par) for par in self.__dict__.items()])
        return parameter_string


class BaseQVAE(AutoEncoderBase):
    """
    Quantum Variational Autoencoder integrated into AutoEncoderBase framework.

    Args:
        input_dimension (int or list of int): Dimensionality of input features.
        activation_fct (callable, optional): Activation function for hidden layers.
        config (object): Configuration object containing:
            - num_latent_units (int)
            - loss_type (str): 'bernoulli' or 'mse'
            - dist_beta (float, default=10.0)
            - kl_beta (float, default=1e-6)
            - weight_decay (float, default=0.01)
        sampler_type (str): Type of sampler for BM ('sa' or 'cim').
        n_batches (int): Number of batches for conditional decoding (0 = no conditioning).
        bm (object, optional): Pre-created Boltzmann Machine. If None, created in create_networks.
        encoder (object, optional): Pre-created encoder. If None, created in create_networks.
        decoder (object, optional): Pre-created decoder. If None, created in create_networks.
        sampler (object, optional): Pre-created sampler. If None, created in create_networks.
        **kwargs: Additional keyword arguments for AutoEncoderBase.
    """
    def __init__(
        self, 
        input_dimension=None, 
        activation_fct=None, 
        config=None,
        sampler_type='sa', 
        n_batches=0, 
        bm=None, 
        encoder=None, 
        decoder=None, 
        sampler=None,
        **kwargs
    ):
        super(BaseQVAE, self).__init__(input_dimension, activation_fct, config, **kwargs)
        self._model_type = "QVAE"   # for identification, can be used in ModelTuner
        self.sampler_type = sampler_type
        self.n_batches = n_batches

        # Parameters from config
        self.dist_beta = getattr(self._config, 'dist_beta', 10.0)
        self.kl_beta = getattr(self._config, 'kl_beta', 1e-6)
        self.weight_decay = getattr(self._config, 'weight_decay', 0.01)

        # Modules (may be passed or created later)
        self.encoder = encoder
        self.decoder = decoder
        self.bm = bm
        self.sampler = sampler
        
        # Bias for Bernoulli reconstruction
        self._train_bias = None

        # Build all networks (subclass must implement create_networks)
        self.create_networks()

    def create_networks(self):
        """Create encoder, decoder, BM and sampler. Subclasses must override."""
        self.encoder = self._create_encoder()
        self.decoder = self._create_decoder()
        self.bm = self._create_bm()
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

    @abc.abstractmethod
    def _create_sampler(self, sampler_type):
        pass

    # -------- Public methods --------
    def set_train_bias(self, mean):
        """Compute train bias from dataset mean for Bernoulli reconstruction."""
        clipped_mean = torch.clamp(mean, 0.001, 0.999).detach()
        self._train_bias = -torch.log(1/clipped_mean - 1)
        return 

    def forward(self, x, batch_idx=None):
        """
        Forward pass through the QVAE.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            batch_idx (torch.Tensor, optional): Batch indices for conditional decoding.
                Required if n_batches > 0.

        Returns:
            tuple: (recon_x, posterior, q, zeta)
                - recon_x: Reconstructed logits (batch_size, input_dim)
                - posterior: Posterior distribution object (MixtureGeneric)
                - q: Encoder output logits (batch_size, latent_dim)
                - zeta: Reparameterized latent sample (batch_size, latent_dim)
        """
        x = x.view(-1, self._input_dimension)

        # For Bernoulli data, we optionally subtract dataset mean
        if self._config.loss_type == 'bernoulli' and self._dataset_mean is not None:
            encoder_x = x - self._dataset_mean
        elif self._config.loss_type == 'mse':
            encoder_x = x
        else:
            raise ValueError(f"Unsupported loss type: {self._config.loss_type}")

        q = self.encoder(encoder_x)  # encoder must implement forward
        posterior, zeta = self.posterior(q, self._config.dist_beta)

        # Conditional decoding based on batch index (if n_batches > 0)
        if self.n_batches > 0:
            if batch_idx is None:
                raise ValueError("batch_idx required when n_batches > 0")
            batch_one_hot = F.one_hot(batch_idx, num_classes=self.n_batches).float().to(zeta.device)
            decoder_input = torch.cat([zeta, batch_one_hot], dim=-1)
        else:
            decoder_input = zeta

        recon_x = self.decoder(decoder_input)  # decoder must implement forward

        # Add Bernoulli bias if needed
        if self._config.loss_type == 'bernoulli' and self._train_bias is not None:
            recon_x = recon_x + self._train_bias.to(recon_x.device)

        return recon_x, posterior, q, zeta

    def loss(self, x, recon_x, posterior, q, zeta):
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
        if self._config.loss_type == 'mse':
            recon_loss = F.mse_loss(
                recon_x, 
                x.view(-1, self._input_dimension), 
                reduction='sum'
            ) / x.size(0)
        elif self._config.loss_type == 'bernoulli':  # bernoulli
            # recon_loss = F.binary_cross_entropy_with_logits(
            #     recon_x, 
            #     x.view(-1, self._input_dimension), 
            #     reduction='sum'
            # ) / x.size(0)
            output_dist = FactorialBernoulliUtil(recon_x)
            recon_loss = -output_dist.log_prob_per_var(x).sum(dim=1).mean()
        else:
            raise ValueError(f"Unsupported loss type: {self._config.loss_type}")
        
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
            if hasattr(self.bm, 'quadratic_coef'):
                loss += bm_weight_decay * torch.sum(self.bm.quadratic_coef ** 2)
            if hasattr(self.bm, 'linear_bias'):
                loss += bm_weight_decay * 0.5 * torch.sum(self.bm.linear_bias ** 2)
        return loss

    def posterior(self, q_logits, beta):
        """Compute posterior distribution and reparameterized sample.

        Args:
            q_logits (torch.Tensor): Encoder output logits (batch_size, latent_dim).
            beta (float): Mixture parameter for MixtureGeneric.

        Returns:
            tuple: (posterior_dist, zeta)
                - posterior_dist: MixtureGeneric object
                - zeta: Reparameterized sample (batch_size, latent_dim)
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
        if hasattr(self.bm, 'quadratic_coef'):
            wd += self.weight_decay * torch.sum(self.bm.quadratic_coef ** 2)
        if hasattr(self.bm, 'linear_bias'):
            wd += self.weight_decay * 0.5 * torch.sum(self.bm.linear_bias ** 2)
        return wd