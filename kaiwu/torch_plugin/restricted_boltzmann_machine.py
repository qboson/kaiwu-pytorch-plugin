# -*- coding: utf-8 -*-
"""受限玻尔兹曼机"""
import torch
import numpy as np
from .abstract_boltzmann_machine import AbstractBoltzmannMachine


class RestrictedBoltzmannMachine(AbstractBoltzmannMachine):
    """创建限制玻尔兹曼机。

    Args:
        num_visible (int): 模型中的可见节点
        num_hidden (int): 模型中的隐藏节点
        h_range (tuple[float, float], optional): 线性权重的范围。
            如果为``None``，使用无限范围。
        j_range (tuple[float, float], optional): 二次权重的范围。
            如果为``None``，使用无限范围。
        device (torch.device, optional): 构造张量的设备。
        如果为``None``，使用CPU。
    """

    def __init__(self, num_visible: int, num_hidden: int, h_range=None, j_range=None):
        super().__init__(h_range=h_range, j_range=j_range)
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.quadratic_coef = torch.nn.Parameter(
            torch.randn((num_visible, num_hidden)) * 0.01
        )
        self.linear_bias = torch.nn.Parameter(torch.zeros(num_hidden + num_visible))

    def clip_parameters(self) -> None:
        """原地裁剪线性和二次偏置权重。"""
        self.get_parameter("linear_bias").data.clamp_(*self.h_range)
        self.get_parameter("quadratic_coef").data.clamp_(*self.j_range)

    def get_hidden(
        self, s_visible: torch.Tensor, requires_grad: bool = False
    ) -> torch.Tensor:
        """将隐藏自旋传播到观测层。
        Args:
            s_visible: 可见层张量
            requires_grad: 是否允许梯度反向传播
        """
        context = torch.enable_grad if requires_grad else torch.no_grad
        with context():
            s_all = torch.zeros(
                s_visible.size(0),
                self.num_hidden + self.num_visible,
                device=self.device,
            )
            s_all[:, : self.num_visible] = s_visible
            prob = torch.sigmoid(
                s_visible @ self.quadratic_coef + self.linear_bias[self.num_visible :]
            )
            s_all[:, self.num_visible :] = prob
            return s_all

    def get_visible(self, s_hidden: torch.Tensor) -> torch.Tensor:
        """将观测自旋传播到隐藏层。"""
        with torch.no_grad():
            s_all = torch.zeros(
                s_hidden.size(0), self.num_hidden + self.num_visible
            ).to(self.device)
            s_all[:, self.num_visible :] = s_hidden
            prob = torch.sigmoid(
                s_hidden @ self.quadratic_coef.t()
                + self.linear_bias[: self.num_visible]
            )
            s_all[:, : self.num_visible] = prob
            return s_all

    def forward(self, s_all: torch.Tensor) -> torch.Tensor:
        """计算哈密顿量。

        Args:
            s_all (torch.tensor): 形状为(B, N)的张量，其中B表示批大小，
                N表示模型中的变量数。

        Returns:
            torch.tensor: 形状为(B,)的哈密顿量。
        """
        tmp = s_all[:, : self.num_visible].matmul(self.quadratic_coef)
        return s_all @ self.linear_bias + torch.sum(
            tmp * s_all[:, self.num_visible :], dim=-1
        )

    def _to_ising_matrix(self):
        """将受限玻尔兹曼机转换为伊辛格式"""
        num_nodes = self.linear_bias.shape[-1]
        with torch.no_grad():
            ising_mat = torch.zeros((num_nodes + 1, num_nodes + 1), device=self.device)
            # 限制玻尔兹曼机：只有可见层和隐藏层之间有连接
            ising_mat[: self.num_visible, self.num_visible : -1] = (
                self.quadratic_coef / 4
            )
            ising_mat[self.num_visible : -1, : self.num_visible] = (
                self.quadratic_coef.T / 4
            )
            ising_bias = self.linear_bias / 2 + ising_mat.sum(dim=0)[:-1]
            ising_mat[:num_nodes, -1] = ising_bias / 2
            ising_mat[-1, :num_nodes] = ising_bias / 2
            return ising_mat.detach().cpu().numpy()
