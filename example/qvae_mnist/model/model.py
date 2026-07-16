# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""
QVAE model implementation for MNIST using BasicEncoder, BasicDecoder and RBM.
"""

from torch import nn

import kaiwu as kw
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import RestrictedBoltzmannMachine, BoltzmannMachine, QVAE
from kaiwu.cim import CIMOptimizer, PrecisionReducer

from .networks import BasicEncoder, BasicDecoder


class MnistQVAE(QVAE):
    """
    Concrete QVAE implementation with BasicEncoder, BasicDecoder and RBM.

    Args:
        input_dimension (int): Input feature dimension.
        activation_fct (callable, optional): Activation function for hidden layers.
        config (object): Configuration object (must contain encoder_hidden_nodes,
            decoder_hidden_nodes, etc.).
        The BM type and sampler type are read from ``config``.
    """
    def __init__(
        self,
        input_dimension,
        activation_fct,
        config,
    ):
        """Initialize the MNIST QVAE from a complete configuration."""
        super().__init__(input_dimension, activation_fct, config)
        self._model_type = "QVAE"

    def _create_encoder(self):
        """
        Create encoder using BasicEncoder.

        Returns:
            BasicEncoder: Encoder network.
        """
        # 根据 config.encoder_hidden_nodes 构造节点序列
        enc_nodes = (
            [self._input_dimension]
            + self.config.encoder_hidden_nodes
            + [self._latent_dimensions]
        )
        node_pairs = [
            (enc_nodes[i], enc_nodes[i+1])
            for i in range(len(enc_nodes)-1)
        ]
        return BasicEncoder(
            node_sequence=node_pairs,
            activation_fct=self._activation_fct,
            weight_decay=self.weight_decay   # 传递衰减系数
        )

    def _create_decoder(self):
        """
        Create decoder using BasicDecoder.

        Returns:
            BasicDecoder: Decoder network (output logits).
        """
        dec_nodes = (
            [self._latent_dimensions]
            + self.config.decoder_hidden_nodes
            + [self._input_dimension]
        )
        node_pairs = [
            (dec_nodes[i], dec_nodes[i+1])
            for i in range(len(dec_nodes)-1)
        ]
        return BasicDecoder(
            node_sequence=node_pairs,
            activation_fct=self._activation_fct,
            output_activation_fct=nn.Identity(),
            weight_decay=self.weight_decay
        )  # 输出 logits

    def _create_bm(self):
        """
        Create RBM with visible and hidden units split from latent dimension.

        Returns:
            RestrictedBoltzmannMachine: RBM instance.
        """
        n_vis = self._latent_dimensions // 2
        n_hid = self._latent_dimensions - n_vis

        bm_type = getattr(self.config, "bm_type", "rbm")
        if bm_type == "rbm":
            bm = RestrictedBoltzmannMachine(num_visible=n_vis, num_hidden=n_hid)
        elif bm_type == "bm":
            bm = BoltzmannMachine(num_nodes=self._latent_dimensions)
        else:
            raise ValueError(f"Unsupported bm type: {bm_type}")
        return bm

    def _create_sampler(self, sampler_type=None):
        """
        Create sampler based on type.

        Args:
            sampler_type (str): 'sa' for simulated annealing, 'cim' for CIM.

        Returns:
            Sampler instance.

        Raises:
            ValueError: If sampler_type is unknown.
        """
        sampler_type = sampler_type or getattr(self.config, "sampler_type", "sa")
        if sampler_type == 'cim':
            kw.common.CheckpointManager.save_dir = './tmp'
            sampler = CIMOptimizer(task_name="qvae_sampling", wait=True)
            sampler = PrecisionReducer(
                sampler,
                precision=8,
                truncated_precision=10,
                target_bits=550,
                only_feasible_solution=False
            )
        elif sampler_type == 'sa':
            sampler = SimulatedAnnealingOptimizer(alpha=0.95)
        else:
            raise ValueError(f"Unsupported sampler type: {sampler_type}")
        return sampler
