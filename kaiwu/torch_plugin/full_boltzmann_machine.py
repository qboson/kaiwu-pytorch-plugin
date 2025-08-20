# -*- coding: utf-8 -*-
"""玻尔兹曼机"""
from typing import Optional, Tuple
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

    def __init__(
        self,
        num_nodes: int,
        h_range: Optional[Tuple[float, float]] = None,
        j_range: Optional[Tuple[float, float]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(h_range=h_range, j_range=j_range, device=device)
        self.num_nodes = num_nodes
        self.quadratic_coef = torch.nn.Parameter(
            torch.randn((self.num_nodes, self.num_nodes), device=self.device) * 0.01
        )
        self.linear_bias = torch.nn.Parameter(
            torch.zeros(self.num_nodes, device=self.device)
        )

    def clip_parameters(self) -> None:
        """原地裁剪线性和二次偏置权重。"""
        self.linear_bias.data.clamp_(*self.h_range)
        self.quadratic_coef.data.clamp_(*self.j_range)
        # 强制对称
        self.quadratic_coef.data = (
            self.quadratic_coef.data + self.quadratic_coef.data.T
        ) / 2.0
        # 将对角线清零
        self.quadratic_coef.data.fill_diagonal_(0)

    def forward(self, s_all: torch.Tensor) -> torch.Tensor:
        """计算哈密顿量。

        Args:
            s_all (torch.tensor): 形状为(B, N)的张量，其中B表示批大小，
                N表示模型中的变量数。

        Returns:
            torch.tensor: 形状为(B,)的哈密顿量。
        """
        linear_term = torch.matmul(s_all, self.linear_bias)
        quadratic_term = 0.5 * torch.einsum(
            "bi,ij,bj->b", s_all, self.quadratic_coef, s_all
        )
        return linear_term + quadratic_term

    def _to_ising_matrix(self):
        """将玻尔兹曼机转换为伊辛矩阵"""
        num_nodes = self.linear_bias.shape[-1]
        ising_mat = torch.zeros((num_nodes + 1, num_nodes + 1), device=self.device)
        ising_mat[:-1, :-1] = self.quadratic_coef / 4.0
        ising_bias = self.linear_bias / 2.0 + torch.sum(ising_mat, axis=0)[:-1]
        ising_mat[:num_nodes, -1] = ising_bias / 2.0
        ising_mat[-1, :num_nodes] = ising_bias / 2.0
        return ising_mat.detach().cpu().numpy()

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

        num_sub_nodes = sub_quadratic.shape[0]
        ising_mat = torch.zeros(
            (num_sub_nodes + 1, num_sub_nodes + 1), device=self.device
        )
        ising_mat[:-1, :-1] = sub_quadratic / 4.0
        ising_bias = sub_linear / 2.0 + torch.sum(ising_mat, axis=0)[:-1]
        ising_mat[:-1, -1] = ising_bias / 2.0
        ising_mat[-1, :-1] = ising_bias / 2.0
        return ising_mat.detach().cpu().numpy()

    def _batch_hidden_to_ising_matrix(self, s_visible_batch: torch.Tensor) -> torch.Tensor:
        """
        给定可见节点的批次，将模型向量化地转换为伊辛格式的子矩阵批次。
        Args:
            s_visible_batch (torch.Tensor): 可见层的状态，形状为(B, num_visible)。
        Returns:
            torch.Tensor: 伊辛格式的子矩阵批次，形状为(B, num_hidden+1, num_hidden+1)。
        """
        B, n_vis = s_visible_batch.shape
        n_hid = self.num_nodes - n_vis

        sub_quadratic = self.quadratic_coef[n_vis:, n_vis:]
        sub_quadratic_vh = (
            self.quadratic_coef[n_vis:, :n_vis]
            + self.quadratic_coef[:n_vis, n_vis:].t()
        )

        # 向量化计算 sub_linear
        # (n_hid, n_vis) @ (B, n_vis).T -> (n_hid, B) -> (B, n_hid)
        sub_linear_batch = (sub_quadratic_vh @ s_visible_batch.t()).t() + self.linear_bias[n_vis:]

        ising_mat_batch = torch.zeros((B, n_hid + 1, n_hid + 1), device=self.device)
        ising_mat_batch[:, :-1, :-1] = sub_quadratic / 4.0

        ising_bias_batch = sub_linear_batch / 2.0 + torch.sum(ising_mat_batch, axis=1)[:, :-1]

        ising_mat_batch[:, :-1, -1] = ising_bias_batch / 2.0
        ising_mat_batch[:, -1, :-1] = ising_bias_batch / 2.0

        return ising_mat_batch

    def _get_conditional_activation(
        self, s_all: torch.Tensor, unit_index: int
    ) -> torch.Tensor:
        """计算给定其他单元时，单个单元的激活值（logit）。"""
        return torch.matmul(s_all, self.quadratic_coef[:, unit_index]) + self.linear_bias[unit_index]

    def gibbs_sample(
        self,
        num_steps: int = 100,
        s_visible: Optional[torch.Tensor] = None,
        num_sample: Optional[int] = None,
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
                    torch.full(
                        (s_visible.size(0), self.num_nodes), 0.5, device=self.device
                    )
                )
                # 将前面可见单元部分替换为给定的可见单元状态
                s_all[:, : s_visible.size(1)] = s_visible.clone()
            else:
                # 如果没有可见单元，全部随机初始化
                s_all = torch.bernoulli(
                    torch.full((num_sample, self.num_nodes), 0.5, device=self.device)
                )

            # 可见单元数量
            n_vis = s_visible.shape[-1] if s_visible is not None else 0
            for _ in range(num_steps):
                # 随机更新顺序（Gibbs采样）
                update_order = torch.randperm(self.num_nodes, device=self.device)
                for unit in update_order:
                    if unit < n_vis:
                        # 跳过可见单元（只采样隐含单元）
                        continue
                    # 计算当前单元的激活值（条件概率的logit）
                    activation = self._get_conditional_activation(s_all, unit.item())
                    # 通过sigmoid得到激活概率
                    prob = torch.sigmoid(activation)
                    # 按概率采样当前单元的状态
                    s_all[:, unit] = torch.bernoulli(prob)
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
        # 1. 批量计算所有 ising 矩阵
        batch_ising_mat_torch = self._batch_hidden_to_ising_matrix(s_visible)
        batch_ising_mat_np = batch_ising_mat_torch.detach().cpu().numpy()

        solutions = []
        for i in range(s_visible.size(0)):
            # 2. 循环调用 sampler
            ising_mat = batch_ising_mat_np[i]
            solution = sampler.solve(ising_mat)

            # 假设最后一列是能量，我们只取自旋状态，并转换为{0,1}
            spins = solution[:, :-1]
            binary_states = (spins + 1) / 2
            
            # 3. 拼接和处理 solution
            visible_part = s_visible[i].unsqueeze(0).expand(binary_states.shape[0], -1)
            hidden_part = torch.FloatTensor(binary_states).to(self.device)
            
            full_solution = torch.cat([visible_part, hidden_part], dim=-1)
            solutions.append(full_solution)

        solutions = torch.cat(solutions, dim=0)
        return solutions
