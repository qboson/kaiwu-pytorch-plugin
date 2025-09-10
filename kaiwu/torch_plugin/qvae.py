# -*- coding: utf-8 -*-
"""quantum variational autoencoder QVAE模型"""

import torch
import numpy as np

from .qvae_dist_util import MixtureGeneric, FactorialBernoulliUtil
from .abstract_boltzmann_machine import AbstractBoltzmannMachine


class QVAE(torch.nn.Module):
    """量子变分自编码器（QVAE）模型

    Args:
        encoder: 编码器模块
        decoder: 解码器模块
        rbm (AbstractBoltzmannMachine): 玻尔兹曼机
        sampler: 采样器
        dist_beta: 分布的beta参数
        train_bias (torch.Tensor): 训练数据的偏置
        is_training (bool): 是否处于训练模式
    """

    def __init__(
        self,
        encoder,
        decoder,
        rbm: AbstractBoltzmannMachine,
        sampler,
        dist_beta,
        mean_x: float,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.rbm = rbm
        self.sampler = sampler
        self.dist_beta = dist_beta
        # 将train_bias转换为PyTorch张量
        self.train_bias = torch.tensor(
            -np.log(1.0 / np.clip(mean_x, 0.001, 0.999) - 1.0).astype(np.float32)
        )
        self.is_training = True

    def posterior(self, q_logits, beta):
        """计算后验分布及其重参数化采样

        Args:
            q_logits (torch.Tensor): 编码器的输出，对数几率
            beta: 分布的beta参数

        Returns:
            tuple: (后验分布对象, 采样结果zeta)
        """
        posterior_dist = MixtureGeneric(q_logits, beta)
        zeta = posterior_dist.reparameterize(self.is_training)
        return posterior_dist, zeta

    def _cross_entropy(
        self, logit_q: torch.Tensor, log_ratio: torch.Tensor
    ) -> torch.Tensor:
        """计算DVAE++中提出的重叠分布的交叉熵项

        Args:
            logit_q (torch.Tensor): 为每个变量定义的伯努利分布的对数几率
            log_ratio (torch.Tensor): 每个ζ的log(r(ζ|z=1)/r(ζ|z=0))

        Returns:
            torch.Tensor: 每个ζ的交叉熵张量
        """
        # 分割logit_q为两部分
        logit_q1 = logit_q[:, : self.num_var1]
        logit_q2 = logit_q[:, self.num_var1 :]

        # 计算概率
        q1 = torch.sigmoid(logit_q1)
        q2 = torch.sigmoid(logit_q2)
        log_ratio1 = log_ratio[:, : self.num_var1]
        q1_pert = torch.sigmoid(logit_q1 + log_ratio1)

        # 计算交叉熵
        cross_entropy = -torch.matmul(
            torch.cat([q1, q2], dim=-1), self.rbm.linear_bias
        ) + -torch.sum(
            torch.matmul(q1_pert, self.rbm.quadratic_coef) * q2, dim=1, keepdim=True
        )
        cross_entropy = cross_entropy.squeeze(dim=1)
        s_neg = self.rbm.sample(self.sampler)
        cross_entropy = cross_entropy - self.rbm(s_neg).mean()
        return cross_entropy

    def _kl_dist_from(self, posterior, post_samples):
        """计算KL散度

        Args:
            posterior: 后验分布对象
            post_samples: 后验分布采样结果

        Returns:
            torch.Tensor: KL散度张量
        """
        entropy = 0
        logit_q = 0
        log_ratio = 0
        entropy += torch.sum(posterior.entropy(), dim=1)

        logit_q = posterior.logit_mu
        log_ratio = posterior.log_ratio(post_samples)
        cross_entropy = self._cross_entropy(logit_q, log_ratio)
        kl = cross_entropy - entropy

        return kl

    def neg_elbo(self, x, kl_beta):
        """计算负ELBO损失

        Args:
            x (torch.Tensor): 输入数据
            kl_beta (float): KL项的权重系数

        Returns:
            tuple: (output, recon_x, neg_elbo, wd_loss, total_kl, cost, q, zeta)
                output: 重构输出（sigmoid激活）
                recon_x: 重构数据
                neg_elbo: 负ELBO损失
                wd_loss: 权重衰减损失
                total_kl: KL散度
                cost: 重构损失
                q: 编码器输出
                zeta: 后验采样结果
        """
        # subtract mean from input
        encoder_x = x - self.train_bias
        recon_x, posterior, q, zeta = self(encoder_x)

        # 添加数据偏置
        recon_x = recon_x + self.train_bias

        output_dist = FactorialBernoulliUtil(recon_x)

        # 经过sigmod
        output = torch.sigmoid(output_dist.logit_mu)

        # 计算KL
        total_kl = self._kl_dist_from(posterior, zeta)
        total_kl = torch.mean(total_kl)
        # expected log prob p(x| z)
        cost = -output_dist.log_prob_per_var(x)  # [256, 784]
        cost = torch.sum(cost, dim=1)  # [256]，每个样本的重构损失
        cost = torch.mean(cost)

        # 计算每个样本的负ELBO，然后取平均
        neg_elbo = total_kl * kl_beta + cost  # 标量

        # weight decay loss
        w_weight_decay = 0.01 * torch.sum(self.rbm.quadratic_coef**2)
        b_weight_decay = 0.005 * torch.sum(self.rbm.linear_bias**2)
        wd_loss = w_weight_decay + b_weight_decay

        return output, recon_x, neg_elbo, wd_loss, total_kl, cost, q, zeta

    def forward(self, x):
        """前向传播

        Args:
            x (torch.Tensor): 输入数据

        Returns:
            tuple: (recon_x, posterior, q, zeta)
                recon_x: 重构数据
                posterior: 后验分布对象
                q: 编码器输出
                zeta: 后验采样结果
        """
        q = self.encoder(x)

        posterior, zeta = self.posterior(q, self.dist_beta)

        recon_x = self.decoder(zeta)

        return recon_x, posterior, q, zeta
