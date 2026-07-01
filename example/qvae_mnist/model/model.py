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
from kaiwu.cim import CIMOptimizer
from kaiwu.preprocess import PrecisionReducer

from .networks import BasicEncoder, BasicDecoder


class MNISTQVAE(QVAE):
    """
    Concrete QVAE implementation with BasicEncoder, BasicDecoder and RBM.

    Args:
        input_dimension (int): Input feature dimension.
        activation_fct (callable, optional): Activation function for hidden layers.
        config (object): Configuration object (must contain encoder_hidden_nodes,
            decoder_hidden_nodes, etc.).
        sampler_type (str, optional): Sampler type ('sa' or 'cim').
    """

    def __init__(
        self,
        input_dimension,
        activation_fct,
        config,
        sampler_type="sa",
    ):
        # QVAE 只保存通用状态，具体模型负责完成组件构建。
        super().__init__(
            input_dimension=input_dimension,
            activation_fct=activation_fct,
            config=config,
            sampler_type=sampler_type,
        )
        self._model_type = "MNISTQVAE"
        self.create_networks()

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
            (enc_nodes[i], enc_nodes[i + 1]) for i in range(len(enc_nodes) - 1)
        ]
        return BasicEncoder(
            node_sequence=node_pairs,
            activation_fct=self._activation_fct,
            weight_decay=self.weight_decay,  # 传递衰减系数
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
            (dec_nodes[i], dec_nodes[i + 1]) for i in range(len(dec_nodes) - 1)
        ]
        return BasicDecoder(
            node_sequence=node_pairs,
            activation_fct=self._activation_fct,
            output_activation_fct=nn.Identity(),
            weight_decay=self.weight_decay,
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

    def _create_sampler(self, sampler_type="sa"):
        """
        Create sampler based on type.

        Args:
            sampler_type (str): 'sa' for simulated annealing, 'cim' for CIM.

        Returns:
            Sampler instance.

        Raises:
            ValueError: If sampler_type is unknown.
        """
        if sampler_type == "cim":
            kw.common.CheckpointManager.save_dir = "./tmp"
            sampler = CIMOptimizer(task_name="qvae_sampling", wait=True)
            sampler = PrecisionReducer(
                sampler,
                precision=8,
                truncated_precision=10,
                target_bits=550,
                only_feasible_solution=False,
            )
        elif sampler_type == "sa":
            sampler = SimulatedAnnealingOptimizer(alpha=0.95)
        else:
            raise ValueError(f"Unsupported sampler type: {sampler_type}")
        return sampler

    def energy(self, x, loss_type):
        """Compute BM energy for MNIST samples in latent space.

        Args:
            x: Input tensor with shape ``(batch_size, input_dimension)``.
            loss_type: Reconstruction loss type. Must match ``config.loss_type``.

        Returns:
            Energy values with shape ``(batch_size,)``.
        """
        if loss_type != self.config.loss_type:
            raise ValueError("loss_type must match model.config.loss_type")

        x = x.view(-1, self._input_dimension)
        if loss_type == "bernoulli":
            encoder_x = x
            if self._dataset_mean is not None:
                encoder_x = encoder_x - x.new_tensor(self._dataset_mean)
        elif loss_type == "mse":
            encoder_x = x
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        q = self.encoder(encoder_x)
        return self.bm((q > 0).float())
