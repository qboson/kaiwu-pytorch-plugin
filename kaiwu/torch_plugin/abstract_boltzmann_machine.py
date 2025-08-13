# -*- coding: utf-8 -*-
"""玻尔兹曼机基类"""
import torch


def clip_parameters_hook(module, *args):  # pylint:disable=unused-argument
    """用于自动裁剪参数的钩子函数。"""
    module.clip_parameters()


class AbstractBoltzmannMachine(torch.nn.Module):
    """玻尔兹曼机的抽象基类。

    Args:
        h_range (tuple[float, float], optional): 线性权重的范围。
            如果为``None``，使用无限范围。
        j_range (tuple[float, float], optional): 二次权重的范围。
            如果为``None``，使用无限范围。
        device (torch.device, optional): 构造张量的设备。
    """

    def __init__(self, h_range=None, j_range=None) -> None:
        super().__init__()
        self.register_buffer(
            "h_range",
            torch.tensor(h_range if h_range is not None else [-torch.inf, torch.inf]),
        )
        self.register_buffer(
            "j_range",
            torch.tensor(j_range if j_range is not None else [-torch.inf, torch.inf]),
        )
        self.register_forward_pre_hook(clip_parameters_hook)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to(self, device=..., dtype=..., non_blocking=...):
        self.device = device
        return super().to(device)

    def forward(self, s_all: torch.Tensor) -> torch.Tensor:
        """计算哈密顿量。

        Args:
            s_all (torch.Tensor): 输入张量

        Returns:
            torch.Tensor: 哈密顿量
        """

    def clip_parameters(self) -> None:
        """原地裁剪线性和二次偏置权重。"""

    def get_ising_matrix(self):
        """将模型转换为伊辛格式。"""
        self.clip_parameters()
        return self._to_ising_matrix()

    def _to_ising_matrix(self):
        """将模型转换为伊辛格式。"""
        raise NotImplementedError("Subclasses must implement _ising method")

    def objective(
        self,
        s_positive: torch.Tensor,
        s_negtive: torch.Tensor,
    ) -> torch.Tensor:
        """一个目标函数，其梯度等价于负对数似然的梯度。

        Args:
            s_positive (torch.Tensor): 观测自旋(数据)的张量，形状为
                (b1, N)，其中b1表示批大小，N表示模型中的变量数。
            s_negtive (torch.Tensor): 从模型中抽取的自旋张量，形状为
                (b2, N)，其中b2表示批大小，N表示模型中的变量数。

        Returns:
            torch.Tensor: 数据和模型平均能量的标量差。
        """
        self.clip_parameters()
        return -(self(s_positive).mean() - self(s_negtive).mean())

    def sample(self, sampler) -> torch.Tensor:
        """从玻尔兹曼机中采样。

        Args:
            sampler: 用于从模型中采样的优化器。采样器可以是kaiwuSDK的CIM或者其他求解器。

        Returns:
            torch.Tensor: 从模型中采样的自旋
        """
        ising_mat = self.get_ising_matrix()
        solution = sampler.solve(ising_mat)
        solution = (solution[:, :-1] + 1) / 2
        solution = torch.FloatTensor(solution)
        solution = solution.to(self.device)
        return solution
