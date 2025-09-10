# -*- coding: utf-8 -*-
"""QVAE模型中使用的分布工具类"""
import torch
import torch.nn.functional as F


class SmoothingDist:
    """平滑分布基类。

    用于定义平滑分布的接口。
    """

    def pdf(self, zeta):
        """概率密度函数 r(zeta|z=0)。

        Args:
            zeta (torch.Tensor): 输入变量。

        Returns:
            torch.Tensor: 概率密度值。
        """
        raise NotImplementedError

    def cdf(self, zeta):
        """累积分布函数 R(zeta|z=0)。

        Args:
            zeta (torch.Tensor): 输入变量。

        Returns:
            torch.Tensor: 累积分布值。
        """
        raise NotImplementedError

    def sample(self, shape):
        """从 r(zeta|z=0) 分布采样。

        Args:
            shape (tuple): 采样形状。

        Returns:
            torch.Tensor: 采样结果。
        """
        raise NotImplementedError

    def log_pdf(self, zeta):
        """对数概率密度 log r(zeta|z=0)。

        Args:
            zeta (torch.Tensor): 输入变量。

        Returns:
            torch.Tensor: 对数概率密度值。
        """
        raise NotImplementedError


class Exponential(SmoothingDist):
    """指数平滑分布类。

    实现了指数平滑分布的 PDF、CDF、采样和对数 PDF 计算。
    """

    def __init__(self, beta):
        """初始化指数分布。

        Args:
            beta (float or torch.Tensor): 指数分布的参数。
        """
        self.beta = torch.tensor(beta, dtype=torch.float32)

    def pdf(self, zeta: torch.Tensor) -> torch.Tensor:
        """概率密度函数。

        Args:
            zeta (torch.Tensor): 输入变量。

        Returns:
            torch.Tensor: 概率密度值。
        """
        return self.beta * torch.exp(-self.beta * zeta) / (1 - torch.exp(-self.beta))

    def cdf(self, zeta: torch.Tensor) -> torch.Tensor:
        """累积分布函数。

        Args:
            zeta (torch.Tensor): 输入变量。

        Returns:
            torch.Tensor: 累积分布值。
        """
        return (1.0 - torch.exp(-self.beta * zeta)) / (1 - torch.exp(-self.beta))

    def sample(self, shape: tuple) -> torch.Tensor:
        """采样。

        Args:
            shape (tuple): 采样形状。

        Returns:
            torch.Tensor: 采样结果。
        """
        rho = torch.rand(shape)
        zeta = -torch.log(1.0 - (1.0 - torch.exp(-self.beta)) * rho) / self.beta
        return zeta

    def log_pdf(self, zeta: torch.Tensor) -> torch.Tensor:
        """对数概率密度。

        Args:
            zeta (torch.Tensor): 输入变量。

        Returns:
            torch.Tensor: 对数概率密度值。
        """
        return (
            torch.log(self.beta)
            - self.beta * zeta
            - torch.log(1 - torch.exp(-self.beta))
        )


class DistUtil:
    """分布工具基类。"""

    def reparameterize(self, is_training):
        """重参数化采样。

        Args:
            is_training (bool): 是否为训练模式。

        Returns:
            torch.Tensor: 采样结果。
        """
        raise NotImplementedError

    def entropy(self):
        """熵计算。

        Returns:
            torch.Tensor: 熵值。
        """
        raise NotImplementedError


def sigmoid_cross_entropy_with_logits(logits, labels):
    """计算 sigmoid 交叉熵损失。

    Args:
        logits (torch.Tensor): 对数几率。
        labels (torch.Tensor): 标签。

    Returns:
        torch.Tensor: sigmoid 交叉熵损失。
    """
    return logits - logits * labels + F.softplus(-logits)


class FactorialBernoulliUtil(DistUtil):
    """阶乘伯努利分布工具类。

    用于处理二值随机变量的概率分布。
    """

    def __init__(self, param):
        """初始化阶乘伯努利分布。

        Args:
            param (torch.Tensor): 分布参数。
        """
        super().__init__()
        self.logit_mu = param

    def reparameterize(self, is_training: bool) -> torch.Tensor:
        """从伯努利分布中采样。

        仅在测试时使用，因为伯努利分布的重参数化在训练时不可微分。

        Args:
            is_training (bool): 是否为训练模式。

        Returns:
            torch.Tensor: 采样结果。

        Raises:
            NotImplementedError: 当 is_training 为 True 时抛出异常。
        """
        if is_training:
            raise NotImplementedError("伯努利分布的重参数化在训练时不可微分")
        device = self.logit_mu.device
        q = torch.sigmoid(self.logit_mu)
        rho = torch.rand_like(q, device=device)
        z = (rho < q).float()
        return z

    def entropy(self):
        """计算伯努利分布的熵。

        Returns:
            torch.Tensor: 熵值。
        """
        mu = torch.sigmoid(self.logit_mu)
        ent = sigmoid_cross_entropy_with_logits(logits=self.logit_mu, labels=mu)
        return ent

    def log_prob_per_var(self, samples):
        """计算样本在分布下的对数概率。

        Args:
            samples (torch.Tensor): 样本矩阵，形状为 (num_samples, num_vars)。

        Returns:
            torch.Tensor: 对数概率矩阵，形状为 (num_samples, num_vars)。
        """
        log_prob = -sigmoid_cross_entropy_with_logits(
            logits=self.logit_mu, labels=samples
        )
        return log_prob


class MixtureGeneric(FactorialBernoulliUtil):
    """混合分布类。

    通过设置定义在 z 分量上的阶乘伯努利分布的对数几率，创建两个重叠分布的混合。
    可与任何继承自 SmoothingDist 类的平滑分布一起工作。
    """

    num_param = 1

    def __init__(self, param, smoothing_dist_beta):
        """初始化混合分布。

        Args:
            param (torch.Tensor): 分布参数。
            smoothing_dist_beta (float): 平滑分布的 beta 参数。
        """
        super().__init__(param)
        self.smoothing_dist = Exponential(smoothing_dist_beta)

    def reparameterize(self, is_training: bool) -> torch.Tensor:
        """使用祖先采样从两个重叠分布的混合中采样。

        使用隐式梯度思想计算样本相对于 logit_q 的梯度。
        该思想在 DVAE# sec 3.4 中提出。
        此函数不实现样本相对于 beta 或平滑变换的其他参数的梯度。

        Args:
            is_training (bool): 是否为训练模式。

        Returns:
            torch.Tensor: 采样结果。
        """
        q = torch.sigmoid(self.logit_mu)

        # 从伯努利分布采样
        z = super().reparameterize(is_training=False)
        shape = z.shape

        # 从平滑分布采样
        zeta = self.smoothing_dist.sample(shape)
        zeta = zeta.to(z.device)

        zeta = torch.where(z == 0.0, zeta, 1.0 - zeta)

        # 计算 PDF 和 CDF
        pdf_0 = self.smoothing_dist.pdf(zeta)
        pdf_1 = self.smoothing_dist.pdf(1.0 - zeta)
        cdf_0 = self.smoothing_dist.cdf(zeta)
        cdf_1 = 1.0 - self.smoothing_dist.cdf(1.0 - zeta)

        # 计算梯度
        grad_q = (cdf_0 - cdf_1) / (q * pdf_1 + (1 - q) * pdf_0)
        grad_q = grad_q.detach()
        grad_term = grad_q * q
        grad_term = grad_term - grad_term.detach()
        # 只让梯度流向 q，而不是 zeta 本身
        zeta = zeta.detach() + grad_term

        return zeta

    def log_prob_per_var(self, samples: torch.Tensor) -> torch.Tensor:
        """计算样本在重叠分布混合下的对数概率。

        Args:
            samples (torch.Tensor): 样本矩阵，形状为 (num_samples, num_vars)。

        Returns:
            torch.Tensor: 对数概率矩阵，形状为 (num_samples, num_vars)。
        """
        q = torch.sigmoid(self.logit_mu)
        pdf_0 = self.smoothing_dist.pdf(samples)
        pdf_1 = self.smoothing_dist.pdf(1.0 - samples)
        log_prob = torch.log(q * pdf_1 + (1 - q) * pdf_0)
        return log_prob

    def log_ratio(self, zeta: torch.Tensor) -> torch.Tensor:
        """计算 KL 梯度所需的 log_ratio（在 DVAE++ 中提出）。

        Args:
            zeta (torch.Tensor): 近似后验样本。

        Returns:
            torch.Tensor: log r(zeta|z=1) - log r(zeta|z=0)。
        """
        log_pdf_0 = self.smoothing_dist.log_pdf(zeta)
        log_pdf_1 = self.smoothing_dist.log_pdf(1.0 - zeta)
        log_ratio = log_pdf_1 - log_pdf_0
        return log_ratio
