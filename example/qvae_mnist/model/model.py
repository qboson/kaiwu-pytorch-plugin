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
from kaiwu.torch_plugin import RestrictedBoltzmannMachine, BoltzmannMachine, BaseQVAE
from kaiwu.cim import CIMOptimizer, PrecisionReducer

from .networks import BasicEncoder, BasicDecoder


class QVAE(BaseQVAE):
    """
    Concrete QVAE implementation with BasicEncoder, BasicDecoder and RBM.

    Args:
        input_dimension (int): Input feature dimension.
        activation_fct (callable, optional): Activation function for hidden layers.
        config (object): Configuration object (must contain encoder_hidden_nodes,
            decoder_hidden_nodes, etc.).
        sampler_type (str, optional): Sampler type ('sa' or 'cim').
        n_batches (int, optional): Number of conditional batches. Default 0.
        **kwargs: Additional kwargs passed to BaseQVAE.
    """
    def __init__(
        self,
        input_dimension,
        activation_fct,
        config,
        # sampler_type="sa",
        # n_batches=0,
        **kwargs,
    ):
        # 调用父类 __init__，父类会调用 create_networks()
        # Optional parameters extracted from kwargs or using defaults
        sampler_type = kwargs.pop('sampler_type', 'sa')
        n_batches = kwargs.pop('n_batches', 0)
        super().__init__(
            input_dimension=input_dimension,
            activation_fct=activation_fct,
            config=config,
            sampler_type=sampler_type,
            n_batches=n_batches,
            **kwargs,
        )
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
            + self._config.encoder_hidden_nodes
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
            + self._config.decoder_hidden_nodes
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
        return RestrictedBoltzmannMachine(num_visible=n_vis, num_hidden=n_hid)

    def _create_sampler(self, sampler_type='sa'):
        """
        Create sampler based on type.

        Args:
            sampler_type (str): 'sa' for simulated annealing, 'cim' for CIM.

        Returns:
            Sampler instance.

        Raises:
            ValueError: If sampler_type is unknown.
        """
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

    # def energy(self, x, loss_type):
    #     """计算 BM 能量（用于评估）"""
    #     x = x.view(-1, self._input_dimension)
    #     if loss_type == 'bernoulli' self._dataset_mean is not None:
    #         x_centered = x - self._dataset_mean
    #     elifl oss_type == 'mse':
    #         x_centered = x
    #     else:
    #         raise ValueError(f"Unsupported loss type: {loss_type}")
    #     q = self.encoder(x_centered)
    #     return self.bm((q > 0).float())
