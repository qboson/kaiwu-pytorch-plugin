import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from dist_util import MixtureGeneric, FactorialBernoulliUtil

from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import RestrictedBoltzmannMachine


class Encoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.weight_decay = weight_decay
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x

    def get_weight_decay(self) -> torch.Tensor:
        """计算权重的L2正则化损失

        对权重矩阵施加L2正则化可以提高模型的泛化能力。

        Returns:
            torch.Tensor: L2正则化损失值
        """
        return self.weight_decay * (
            torch.sum(self.fc1.weight**2) + torch.sum(self.fc2.weight**2)
        )


class Decoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.weight_decay = weight_decay
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z):
        z = self.fc1(z)
        z = self.norm1(z)
        z = F.tanh(z)
        z = self.fc2(z)

        return z

    def get_weight_decay(self) -> torch.Tensor:
        """计算权重的L2正则化损失

        对权重矩阵施加L2正则化可以提高模型的泛化能力。

        Returns:
            torch.Tensor: L2正则化损失值
        """
        return self.weight_decay * (
            torch.sum(self.fc1.weight**2) + torch.sum(self.fc2.weight**2)
        )


class QVAE(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_var1: int,
        num_var2: int,
        dist_beta: float,
        mean_x: float,
    ) -> None:
        super().__init__()

        # 参数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dist_beta = dist_beta
        self.num_var1 = num_var1
        self.num_var2 = num_var2
        # 将train_bias转换为PyTorch张量
        self.train_bias = torch.tensor(
            -np.log(1.0 / np.clip(mean_x, 0.001, 0.999) - 1.0).astype(np.float32)
        )
        self.is_training = True

        # 网络
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, weight_decay=0.01)
        self.rbm = RestrictedBoltzmannMachine(
            num_visible=num_var1, num_hidden=num_var2, h_range=[-1, 1], j_range=[-1, 1]
        )
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, weight_decay=0.01)

        # 添加kaiwu的license信息
        self.sampler = SimulatedAnnealingOptimizer(alpha=0.95)

    def to(self, device):
        """重写to方法，确保所有组件都移动到正确的设备上"""
        super().to(device)
        self.rbm.device = device
        self.train_bias = self.train_bias.to(device)
        return self

    def posterior(self, q_logits, beta):
        """
        Args:
            q_logits (torch.Tensor): encoder的输出，对数几率
        """

        posterior_dist = MixtureGeneric(q_logits, beta)
        zeta = posterior_dist.reparameterize(self.is_training)
        return posterior_dist, zeta

    def cross_entropy(
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

    def kl_dist_from(self, posterior, post_samples, is_training=True):

        entropy = 0
        logit_q = 0
        log_ratio = 0
        entropy += torch.sum(posterior.entropy(), dim=1)

        logit_q = posterior.logit_mu
        log_ratio = posterior.log_ratio(post_samples)
        cross_entropy = self.cross_entropy(logit_q, log_ratio)
        kl = cross_entropy  # - entropy

        return kl

    def neg_elbo(self, x, kl_beta):
        # subtract mean from input
        encoder_x = x - self.train_bias
        recon_x, posterior, q, zeta = self.forward(encoder_x)

        # 添加数据偏置
        recon_x = recon_x + self.train_bias

        output_dist = FactorialBernoulliUtil(recon_x)

        # 经过sigmod
        output = torch.sigmoid(output_dist.logit_mu)

        # 计算KL
        total_kl = self.kl_dist_from(posterior, zeta, self.is_training)
        total_kl = torch.mean(total_kl)
        # expected log prob p(x| z)
        cost = -output_dist.log_prob_per_var(x)  # [256, 784]
        cost = torch.sum(cost, dim=1)  # [256]，每个样本的重构损失
        cost = torch.mean(cost)

        # 计算每个样本的负ELBO，然后取平均
        neg_elbo = total_kl * kl_beta + cost  # 标量

        enc_wd_loss = self.encoder.get_weight_decay()
        dec_wd_loss = self.decoder.get_weight_decay()
        w_weight_decay = 0.01 * torch.sum(self.rbm.quadratic_coef**2)
        b_weight_decay = 0.005 * torch.sum(self.rbm.linear_bias**2)
        wd_loss = enc_wd_loss + dec_wd_loss + w_weight_decay + b_weight_decay

        return output, recon_x, neg_elbo, wd_loss, total_kl, cost, q, zeta

    def forward(self, x):
        q = self.encoder(x)

        posterior, zeta = self.posterior(q, self.dist_beta)

        recon_x = self.decoder(zeta)

        return recon_x, posterior, q, zeta
