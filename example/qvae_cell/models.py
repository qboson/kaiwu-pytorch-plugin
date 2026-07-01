import torch
import torch.nn as nn
import torch.nn.functional as F
from kaiwu.torch_plugin import QVAE


class QVAEEncoder(nn.Module):
    """将单细胞表达矩阵编码为 QVAE 潜变量 logits。

    Args:
        input_dim: 输入基因特征数。
        hidden_dim: 隐藏层维度。
        latent_dim: 潜变量维度。
        normalization_method: 归一化方式，可选 ``layer`` 或 ``batch``。
    """

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
        """执行编码器前向传播。

        Args:
            x: 形状为 ``(batch_size, input_dim)`` 的表达矩阵。

        Returns:
            形状为 ``(batch_size, latent_dim)`` 的潜变量 logits。
        """
        h = self.fc1(x)
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        # 输出 q logits，后续由 QVAE 的 posterior 转成离散潜变量分布。
        return self.fc2(h)


class QVAEDecoder(nn.Module):
    """从 QVAE 潜变量重构单细胞表达矩阵。

    Args:
        latent_dim: 解码器输入维度，包含潜变量和可选 batch one-hot。
        hidden_dim: 隐藏层维度。
        output_dim: 输出基因特征数。
        normalization_method: 归一化方式，可选 ``layer`` 或 ``batch``。
    """

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
        """执行解码器前向传播。

        Args:
            zeta: 潜变量及可选 batch 条件组成的输入张量。

        Returns:
            重构表达值或 Bernoulli logits。
        """
        h = self.fc1(zeta)
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        # 返回重构 logits / 连续重构值；具体解释取决于 loss_type。
        return self.fc2(h)


class CellQVAE(QVAE):
    """面向单细胞数据并支持 batch 条件解码的 QVAE。

    Args:
        input_dimension: 输入基因特征数。
        activation_fct: 网络使用的激活函数。
        config: QVAE 配置，包含潜变量维度、损失类型和损失权重。
        encoder: 单细胞编码器。
        decoder: 单细胞解码器。
        bm: 玻尔兹曼机先验。
        sampler: BM 负相采样器。
        sampler_type: 采样器类型标识。
        n_batches: 数据批次数；大于 0 时向解码器追加 batch one-hot。
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
        sampler_type="sa",
        n_batches=0,
    ):
        super().__init__(
            input_dimension=input_dimension,
            activation_fct=activation_fct,
            config=config,
            encoder=encoder,
            decoder=decoder,
            bm=bm,
            sampler=sampler,
            sampler_type=sampler_type,
        )
        self.n_batches = n_batches
        self._model_type = "CellQVAE"

    def _decoder_input(self, zeta, batch_idx=None):
        """构造包含 batch 条件的解码器输入。

        Args:
            zeta: QVAE 潜变量。
            batch_idx: 每个样本的 batch 整数索引。

        Returns:
            原始潜变量，或拼接 batch one-hot 后的张量。

        Raises:
            ValueError: 配置了 batch 条件但没有传入 ``batch_idx``。
        """
        # 若存在 batch 信息，把 one-hot batch 拼到潜变量后面，帮助 decoder 消化批次差异。
        if self.n_batches <= 0:
            return zeta
        if batch_idx is None:
            raise ValueError("batch_idx is required when n_batches > 0")
        batch_one_hot = (
            F.one_hot(batch_idx, num_classes=self.n_batches).float().to(zeta.device)
        )
        return torch.cat([zeta, batch_one_hot], dim=-1)

    def _encode(self, x):
        """按照配置的重构损失编码输入。

        Args:
            x: 单细胞表达矩阵。

        Returns:
            ``(q, posterior, zeta)``，分别为编码器 logits、后验分布和潜变量样本。

        Raises:
            ValueError: 配置了不支持的损失类型。
        """
        x = x.view(-1, self._input_dimension)
        if self.config.loss_type == "mse":
            encoder_x = x
        elif self.config.loss_type == "bernoulli":
            encoder_x = x
            if self._dataset_mean is not None:
                encoder_x = encoder_x - torch.as_tensor(
                    self._dataset_mean,
                    dtype=x.dtype,
                    device=x.device,
                )
        else:
            raise ValueError(f"Unsupported loss type: {self.config.loss_type}")
        q = self.encoder(encoder_x)
        posterior, zeta = self.posterior(q, self.config.dist_beta)
        return q, posterior, zeta

    def forward(self, x, batch_idx=None):
        """执行带可选 batch 条件的 QVAE 前向传播。

        Args:
            x: 单细胞表达矩阵。
            batch_idx: 每个样本的 batch 整数索引。

        Returns:
            ``(recon_x, posterior, q, zeta)``，分别为重构结果、后验分布、
            编码器 logits 和潜变量样本。
        """
        q, posterior, zeta = self._encode(x)
        recon_x = self.decoder(self._decoder_input(zeta, batch_idx))
        if self.config.loss_type == "bernoulli":
            recon_x = recon_x + self._train_bias
        return recon_x, posterior, q, zeta

    def energy(self, x, loss_type):
        """计算单细胞样本在 BM 潜空间中的能量。

        Args:
            x: 单细胞表达矩阵。
            loss_type: 损失类型，必须与模型配置一致。

        Returns:
            每个样本对应的 BM 能量。

        Raises:
            ValueError: ``loss_type`` 与模型配置不一致。
        """
        if loss_type != self.config.loss_type:
            raise ValueError("loss_type must match model.config.loss_type")
        q, _, _ = self._encode(x)
        return self.bm((q > 0).float())
