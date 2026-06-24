# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""
QVAE model implementation for MNIST using BasicEncoder, BasicDecoder and RBM.
"""

from torch import nn
import torch.nn.functional as F

import kaiwu as kw
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import RestrictedBoltzmannMachine, BoltzmannMachine, BaseQVAE
from kaiwu.torch_plugin.qvae_dist_util import MixtureGeneric, FactorialBernoulliUtil
from kaiwu.cim import CIMOptimizer, PrecisionReducer

class QVAEEncoder(nn.Module):
    """单细胞表达矩阵到 QVAE 潜变量 logits 的编码器。"""

    def __init__(self, input_dim, hidden_dim, latent_dim, normalization_method="layer"):
        super().__init__()
        # 这里保持一个浅层 MLP，避免在小样本细胞数据上引入过强的模型复杂度。
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if normalization_method == "layer":
            self.norm = nn.LayerNorm(hidden_dim)
        elif normalization_method == "batch":
            self.norm = nn.BatchNorm1d(hidden_dim)
        else:
            raise ValueError("normalization_method must be 'layer' or 'batch'")
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.fc1(x)
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        # 输出 q logits，后续由 QVAE 的 posterior 转成离散潜变量分布。
        return self.fc2(h)


class QVAEDecoder(nn.Module):
    """从 QVAE 潜变量重构输入表达矩阵的解码器。"""

    def __init__(
        self, latent_dim, hidden_dim, output_dim, normalization_method="layer"
    ):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        if normalization_method == "layer":
            self.norm = nn.LayerNorm(hidden_dim)
        elif normalization_method == "batch":
            self.norm = nn.BatchNorm1d(hidden_dim)
        else:
            raise ValueError("normalization_method must be 'layer' or 'batch'")
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, zeta):
        h = self.fc1(zeta)
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        # 返回重构 logits / 连续重构值；具体解释取决于 loss_type。
        return self.fc2(h)


class CellQVAE(BaseQVAE):
    """面向单细胞数据的 QVAE，支持 batch 条件与两阶段训练。"""
    
    def __init__(
            self, 
            input_dimension, 
            activation_fct, 
            config, 
            # n_batches=0, 
            **kwargs
    ):
        sampler_type = kwargs.pop('sampler_type', 'sa')
        n_batches = kwargs.pop('n_batches', 0)
        super().__init__(input_dimension, activation_fct, config, sampler_type=sampler_type, n_batches=n_batches, **kwargs)
        self._model_type = "CellQVAE"
        # self.n_batches = n_batches
        self.bm = self._create_bm(bm_type='bm')
        # self.sampler = self._create_sampler(self.sampler_type)

        # 替换 encoder/decoder 为自定义结构
        self.encoder = self._create_custom_encoder()
        self.decoder = self._create_custom_decoder()
        # 保存 batch 索引供 forward 使用

    # def _create_bm(self):
    #     """创建玻尔兹曼机 (BM)"""
    #     # 使用 RBM 或全连接 BM，这里沿用原 QVAE 的设计
    #     return BoltzmannMachine(num_nodes=self._latent_dimensions)
    def _create_bm(self, bm_type='rbm'):
        """
        Create BM with latent dimension.

        Returns:
            RestrictedBoltzmannMachine: RBM instance.
        """
        n_vis = self._latent_dimensions // 2
        n_hid = self._latent_dimensions - n_vis

        if bm_type =="rbm":
            bm = RestrictedBoltzmannMachine(num_visible=n_vis, num_hidden=n_hid)
        elif bm_type =="bm":
            bm = BoltzmannMachine(num_nodes=self._latent_dimensions)
        else:
            raise ValueError(f"Unsupported bm type: {bm_type}")
        return bm

    def _create_sampler(self, sampler_type):
        """创建采样器 (SA 或 CIM)"""
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
            return sampler
        elif sampler_type == 'sa':
            return SimulatedAnnealingOptimizer(alpha=0.95)
        else:
            raise ValueError(f"Unsupported sampler type: {sampler_type}")

    def _create_custom_encoder(self):
        """重写父类方法，返回自定义编码器"""
        return QVAEEncoder(
            input_dim=self._input_dimension,
            hidden_dim=self._config.hidden_dim,
            latent_dim=self._latent_dimensions,
            normalization_method=getattr(self._config, 'normalization_method', 'layer')
        )

    def _create_custom_decoder(self):
        """重写父类方法，返回自定义解码器，输入维度为 latent_dim + n_batches"""
        decoder_input_dim = self._latent_dimensions + self.n_batches
        return QVAEDecoder(
            latent_dim=decoder_input_dim,
            hidden_dim=self._config.hidden_dim,
            output_dim=self._input_dimension,
            normalization_method=getattr(self._config, 'normalization_method', 'layer')
        )

    # # ------ 两阶段训练方法 ------
    # def cell_loss(self, x, batch_idx, loss_type, kl_beta):
    #     """用于第一阶段 VAE 参数更新的损失（返回 total_loss, kl, recon_loss）"""
    #     recon_logits, posterior, q, zeta = self.forward(x, batch_idx)
    #     if loss_type == 'mse':
    #         recon_loss = F.mse_loss(recon_logits, x.view(-1, self._input_dimension), reduction='sum') / x.size(0)
    #     else:  # bernoulli
    #         output_dist = FactorialBernoulliUtil(recon_x)
    #         recon_loss = -output_dist.log_prob_per_var(x.view(-1, self._input_dimension)).sum(dim=1).mean()
    #     kl_loss = self._kl_dist_from(posterior).mean()
    #     total_loss = recon_loss + kl_beta * kl_loss
    #     return total_loss, kl_loss, recon_loss, q

    # def bm_phase_loss(self, q, bm_weight_decay=0.0):
    #     """用于第二阶段 BM 参数更新的损失"""
    #     # 使用硬阈值采样作为正样本
    #     positive_state = (q.detach() > 0).float()
    #     loss = self.bm.objective(positive_state, self.bm.sample(self.sampler))
    #     if bm_weight_decay > 0:
    #         loss += bm_weight_decay * (torch.sum(self.bm.quadratic_coef**2) + torch.sum(self.bm.linear_bias**2))
    #     return loss

    # def energy(self, x, loss_type):
    #     """计算 BM 能量（用于评估）"""
    #     x = x.view(-1, self._input_dimension)
    #     if self._train_bias is not None:
    #         x_centered = x - self._dataset_mean
    #     else:
    #         x_centered = x
    #     q = self.encoder(x_centered)
    #     return self.bm((q > 0).float())
