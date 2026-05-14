import torch
import torch.nn as nn
import torch.nn.functional as F
from kaiwu.torch_plugin import QVAE
from kaiwu.torch_plugin.qvae_dist_util import FactorialBernoulliUtil


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


class CellQVAE(QVAE):
    """面向单细胞数据的 QVAE 封装，额外支持 batch 条件输入和两类重构损失。"""

    def __init__(self, *args, n_batches=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_batches = n_batches

    def _decoder_input(self, zeta, batch_idx=None):
        # 若存在 batch 信息，把 one-hot batch 拼到潜变量后面，帮助 decoder 消化批次差异。
        if self.n_batches <= 0:
            return zeta
        if batch_idx is None:
            raise ValueError("batch_idx is required when n_batches > 0")
        batch_one_hot = (
            F.one_hot(batch_idx, num_classes=self.n_batches).float().to(zeta.device)
        )
        return torch.cat([zeta, batch_one_hot], dim=-1)

    def encode_for_loss(self, x, loss_type):
        # Bernoulli 重构使用训练集均值偏置做中心化；MSE 直接拟合原始连续表达值。
        encoder_x = x if loss_type == "mse" else x - self.train_bias
        q = self.encoder(encoder_x)
        posterior, zeta = self.posterior(q, self.dist_beta)
        return q, posterior, zeta

    def mse_elbo(self, x, batch_idx, kl_beta):
        # MSE 版本的 ELBO：连续重构误差 + 加权 KL 正则。
        q, posterior, zeta = self.encode_for_loss(x, "mse")
        recon_x = self.decoder(self._decoder_input(zeta, batch_idx))
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
        kl = self._kl_dist_from(posterior).mean()
        return recon_loss + kl_beta * kl, kl, recon_loss, q

    def bernoulli_elbo(self, x, batch_idx, kl_beta):
        # encoder_x = x - self.train_bias
        # Bernoulli 版本适合已归一化到 [0, 1] 的输入，decoder 输出作为 Bernoulli logits。
        q, posterior, zeta = self.encode_for_loss(x, "bernoulli")
        recon_x = self.decoder(self._decoder_input(zeta, batch_idx)) + self.train_bias
        output_dist = FactorialBernoulliUtil(recon_x)
        kl = self._kl_dist_from(posterior).mean()
        recon_loss = -output_dist.log_prob_per_var(x).sum(dim=1).mean()
        return recon_loss + kl_beta * kl, kl, recon_loss, q

    def cell_loss(self, x, batch_idx, loss_type, kl_beta):
        # 统一训练入口，方便 Trainer 不关心具体重构分布。
        if loss_type == "mse":
            return self.mse_elbo(x, batch_idx, kl_beta)
        return self.bernoulli_elbo(x, batch_idx, kl_beta)

    def bm_phase_loss(self, q, bm_weight_decay=0.0):
        # BM 阶段用 encoder 的 q 构造正样本，再和采样器得到的负样本做能量目标。
        positive_state = (q.detach() > 0).float()
        loss = self.bm.objective(positive_state, self.bm.sample(self.sampler))
        # 使用 sigmoid(q) 的软状态作为正相，可减少硬阈值带来的不稳定。
        loss = self.bm.objective(
            torch.sigmoid(q.detach()), self.bm.sample(self.sampler)
        )
        if bm_weight_decay > 0:
            loss = loss + bm_weight_decay * (
                torch.sum(self.bm.quadratic_coef**2) + torch.sum(self.bm.linear_bias**2)
            )
        return loss

    def energy(self, x, loss_type):
        # 评估阶段把样本映射到离散 latent 状态，并返回 BM 能量用于可视化/分析。
        q, _, _ = self.encode_for_loss(x, loss_type)
        return self.bm((q > 0).float())
