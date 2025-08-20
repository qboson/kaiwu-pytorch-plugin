# -*- coding: utf-8 -*-
"""受限玻尔兹曼机"""
from typing import Optional, Tuple
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

    def __init__(
        self,
        num_visible: int,
        num_hidden: int,
        h_range: Optional[Tuple[float, float]] = None,
        j_range: Optional[Tuple[float, float]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(h_range=h_range, j_range=j_range, device=device)
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        # 使用更标准的名称
        self.weights = torch.nn.Parameter(
            torch.randn((num_visible, num_hidden), device=self.device) * 0.01
        )
        self.visible_bias = torch.nn.Parameter(
            torch.zeros(num_visible, device=self.device)
        )
        self.hidden_bias = torch.nn.Parameter(
            torch.zeros(num_hidden, device=self.device)
        )

    def clip_parameters(self) -> None:
        """原地裁剪线性和二次偏置权重。"""
        self.visible_bias.data.clamp_(*self.h_range)
        self.hidden_bias.data.clamp_(*self.h_range)
        self.weights.data.clamp_(*self.j_range)

    def prob_h_given_v(self, v: torch.Tensor) -> torch.Tensor:
        """计算给定可见层时，隐藏层单元的激活概率 P(h=1|v)。"""
        return torch.sigmoid(v @ self.weights + self.hidden_bias)

    def sample_h_given_v(self, v: torch.Tensor) -> torch.Tensor:
        """从 P(h|v) 中采样隐藏层状态。"""
        prob = self.prob_h_given_v(v)
        return torch.bernoulli(prob)

    def prob_v_given_h(self, h: torch.Tensor) -> torch.Tensor:
        """计算给定隐藏层时，可见层单元的激活概率 P(v=1|h)。"""
        return torch.sigmoid(h @ self.weights.t() + self.visible_bias)

    def sample_v_given_h(self, h: torch.Tensor) -> torch.Tensor:
        """从 P(v|h) 中采样可见层状态。"""
        prob = self.prob_v_given_h(h)
        return torch.bernoulli(prob)

    def forward(self, s_all: torch.Tensor) -> torch.Tensor:
        """计算哈密顿量(能量)。
        E(v, h) = -v*a - h*b - v*W*h

        Args:
            s_all (torch.tensor): 形状为(B, N_vis + N_hid)的张量。

        Returns:
            torch.tensor: 形状为(B,)的能量。
        """
        s_visible = s_all[:, : self.num_visible]
        s_hidden = s_all[:, self.num_visible :]

        linear_term = (s_visible @ self.visible_bias) + (s_hidden @ self.hidden_bias)
        quadratic_term = torch.einsum("bi,ij,bj->b", s_visible, self.weights, s_hidden)
        return -linear_term - quadratic_term

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """计算可见层 v 的自由能。
        F(v) = -v*a - sum_j log(1 + exp(v*W_j + c_j))
        """
        v_bias_term = torch.matmul(v, self.visible_bias)
        wx_b = torch.matmul(v, self.weights) + self.hidden_bias
        hidden_term = torch.sum(torch.nn.functional.softplus(wx_b), dim=1)
        return -v_bias_term - hidden_term

    def _to_ising_matrix(self):
        """将受限玻尔兹曼机转换为伊辛格式"""
        num_nodes = self.num_visible + self.num_hidden
        ising_mat = torch.zeros((num_nodes + 1, num_nodes + 1), device=self.device)

        quad_coef_div4 = self.weights / 4.0

        ising_mat[: self.num_visible, self.num_visible : -1] = quad_coef_div4
        ising_mat[self.num_visible : -1, : self.num_visible] = quad_coef_div4.T

        linear_bias = torch.cat((self.visible_bias, self.hidden_bias))
        ising_bias = linear_bias / 2.0 + torch.sum(ising_mat, dim=0)[:-1]

        ising_mat[:num_nodes, -1] = ising_bias / 2.0
        ising_mat[-1, :num_nodes] = ising_bias / 2.0

        return ising_mat.detach().cpu().numpy()
