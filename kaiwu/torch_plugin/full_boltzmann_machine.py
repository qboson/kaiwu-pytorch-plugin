# -*- coding: utf-8 -*-
"""玻尔兹曼机"""
import torch
import numpy as np
from .restricted_boltzmann_machine import AbstractBoltzmannMachine


class BoltzmannMachine(AbstractBoltzmannMachine):
    """创建玻尔兹曼机。

    Args:
        num_nodes (int): 模型中的节点总数
        h_range (tuple[float, float], optional): 线性权重的范围。
            如果为``None``，使用无限范围。
        j_range (tuple[float, float], optional): 二次权重的范围。
            如果为``None``，使用无限范围。
        device (torch.device, optional): 构造张量的设备。
        如果为``None``，使用CPU。
    """

    def __init__(self, num_nodes: int, h_range=None, j_range=None):
        super().__init__(h_range=h_range, j_range=j_range)
        self.num_nodes = num_nodes
        self.quadratic_coef = torch.nn.Parameter(
            torch.randn((self.num_nodes, self.num_nodes)) * 0.01
        )
        self.linear_bias = torch.nn.Parameter(torch.zeros(self.num_nodes))

    def clip_parameters(self) -> None:
        """原地裁剪线性和二次偏置权重。"""
        self.get_parameter("linear_bias").data.clamp_(*self.h_range)
        self.get_parameter("quadratic_coef").data.clamp_(*self.j_range)

    def forward(self, s_all: torch.Tensor) -> torch.Tensor:
        """计算哈密顿量。

        Args:
            s_all (torch.tensor): 形状为(B, N)的张量，其中B表示批大小，
                N表示模型中的变量数。

        Returns:
            torch.tensor: 形状为(B,)的哈密顿量。
        """
        tmp = s_all.matmul(self.quadratic_coef)
        return s_all @ self.linear_bias + torch.sum(tmp * s_all, dim=-1)

    def _to_ising_matrix(self):
        """将玻尔兹曼机转换为伊辛矩阵"""
        linear_bias = self.linear_bias.detach().cpu().numpy()
        quadratic_coef = self.quadratic_coef.detach().cpu().numpy()
        num_nodes = self.linear_bias.shape[-1]

        ising_mat = np.zeros((num_nodes + 1, num_nodes + 1))
        ising_mat[:-1, :-1] = quadratic_coef / 4
        ising_bias = linear_bias / 2 + np.sum(ising_mat, axis=0)[:-1]
        ising_mat[:num_nodes, -1] = ising_bias / 2
        ising_mat[-1, :num_nodes] = ising_bias / 2
        return ising_mat

    def _hidden_to_ising_matrix(self, s_visible: torch.Tensor) -> np.ndarray:
        """给定可见节点的情况，将模型转换为伊辛格式的子矩阵。
        Args:
            s_visible (torch.Tensor): 可见层的状态，形状为(B, num_visible)。
        Returns:
            np.ndarray: 伊辛格式的子矩阵。
        """
        n_vis = s_visible.shape[-1]
        sub_quadratic = self.quadratic_coef[n_vis:, n_vis:]
        sub_quadratic_vh = (
            self.quadratic_coef[n_vis:, :n_vis]
            + self.quadratic_coef[:n_vis, n_vis:].t()
        )
        sub_linear = sub_quadratic_vh @ s_visible + self.linear_bias[n_vis:]
        sub_linear = sub_linear.detach().cpu().numpy()
        sub_quadratic = sub_quadratic.detach().cpu().numpy()

        ising_mat = np.zeros((sub_quadratic.shape[0] + 1, sub_quadratic.shape[0] + 1))
        ising_mat[:-1, :-1] = sub_quadratic / 4
        ising_bias = sub_linear / 2 + np.sum(ising_mat, axis=0)[:-1]
        ising_mat[:-1, -1] = ising_bias / 2
        ising_mat[-1, :-1] = ising_bias / 2
        return ising_mat

    def gibbs_sample(
        self, num_steps: int = 100, s_visible: torch.Tensor = None, num_sample=None
    ) -> torch.Tensor:
        """从玻尔兹曼机中采样。
        Args:
            num_steps (int): Gibbs采样的步数。
            s_visible (torch.Tensor, optional): 可见层的状态，
                形状为(B, num_visible)。如果为``None``，则随机初始化可见层。
            num_sample (int, optional): 采样的数量。
                如果为``None``，则使用s_visible的批大小。
        """
        with torch.no_grad():
            # 初始化：如果没有提供可见单元状态且没有指定采样数，则报错
            if s_visible is None and num_sample is None:
                raise ValueError("Either s_visible or num_sample must be provided.")
            # 如果没有指定采样数，则用可见单元的 batch size
            num_sample = s_visible.size(0) if num_sample is None else num_sample
            if s_visible is not None:
                # 初始化所有单元（可见+隐含）为0.5概率的伯努利分布
                s_all = torch.bernoulli(
                    torch.full((s_visible.size(0), self.num_nodes), 0.5)
                )
                # 将前面可见单元部分替换为给定的可见单元状态
                s_all[:, : s_visible.size(1)] = s_visible.clone()
            else:
                # 如果没有可见单元，全部随机初始化
                s_all = torch.bernoulli(torch.full((num_sample, self.num_nodes), 0.5))

            # 可见单元数量
            n_vis = s_visible.shape[-1] if s_visible is not None else 0
            for _ in range(num_steps):
                # 随机更新顺序（Gibbs采样）
                update_order = torch.randperm(self.num_nodes)
                for unit in update_order:
                    if unit < n_vis:
                        # 跳过可见单元（只采样隐含单元）
                        continue
                    # 计算当前单元的激活值（条件概率的logit）
                    activation = (
                        torch.matmul(s_all, self.quadratic_coef[:, unit])
                        + self.linear_bias[unit]
                    )
                    # 通过sigmoid得到激活概率
                    prob = torch.sigmoid(activation)
                    # 按概率采样当前单元的状态
                    s_all[:, unit] = (prob > torch.rand_like(prob)).float()
            # 返回采样后的所有单元状态
            return s_all

    def condition_sample(self, sampler, s_visible) -> torch.Tensor:
        """给定部分节点后，从玻尔兹曼机中采样。

        Args:
            sampler: 用于从模型中采样的优化器。
            s_visible: 可见层状态。

        Returns:
            torch.Tensor: 从模型中采样的自旋
                (形状由``sampler``和``sample_params``规定)。
        """
        solutions = []
        for i in range(s_visible.size(0)):
            ising_mat = self._hidden_to_ising_matrix(s_visible[i])
            solution = sampler.solve(ising_mat)
            solution = (solution[:, :-1] + 1) / 2
            solution = torch.cat(
                [
                    s_visible[i].unsqueeze(0).expand(solution.shape[0], -1),
                    torch.FloatTensor(solution),
                ],
                dim=-1,
            )
            solutions.append(solution)
        solutions = torch.cat(solutions, dim=0)
        solutions = torch.FloatTensor(solutions)
        solutions = solutions.to(self.device)
        return solutions
