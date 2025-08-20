# -*- coding: utf-8 -*-
"""玻尔兹曼机基类"""
import abc
from typing import Optional, Tuple, Protocol, runtime_checkable, Any
import numpy as np
import torch


@runtime_checkable
class Sampler(Protocol):
    """定义采样器必须实现的接口协议。"""

    def solve(self, ising_matrix: np.ndarray) -> np.ndarray:
        """根据伊辛矩阵求解并返回结果。"""
        ...


def clip_parameters_hook(module, *args):  # pylint:disable=unused-argument
    """用于自动裁剪参数的钩子函数。"""
    module.clip_parameters()


class AbstractBoltzmannMachine(torch.nn.Module, abc.ABC):
    """玻尔兹曼机的抽象基类。

    Args:
        h_range (tuple[float, float], optional): 线性权重的范围。
            如果为``None``，使用无限范围。
        j_range (tuple[float, float], optional): 二次权重的范围。
            如果为``None``，使用无限范围。
        device (torch.device, optional): 构造张量的设备。
    """

    def __init__(
        self,
        h_range: Optional[Tuple[float, float]] = None,
        j_range: Optional[Tuple[float, float]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.register_buffer(
            "h_range",
            torch.tensor(
                h_range if h_range is not None else [-torch.inf, torch.inf],
                device=self.device,
            ),
        )
        self.register_buffer(
            "j_range",
            torch.tensor(
                j_range if j_range is not None else [-torch.inf, torch.inf],
                device=self.device,
            ),
        )
        self.register_forward_pre_hook(clip_parameters_hook)
        self.to(self.device)

    def to(self, *args: Any, **kwargs: Any) -> "AbstractBoltzmannMachine":
        """将模型移动到指定设备。"""
        # 从参数中解析 device
        # .to(device) -> args[0]
        # .to(device=device) -> kwargs['device']
        if "device" in kwargs:
            self.device = kwargs["device"]
        elif len(args) > 0 and isinstance(args[0], (torch.device, str)):
            self.device = torch.device(args[0])

        # 调用父类的方法，并返回 self
        return super().to(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, s_all: torch.Tensor) -> torch.Tensor:
        """计算哈密顿量。
        
        Args:
            s_all (torch.Tensor): 输入张量
            
        Returns:
            torch.Tensor: 哈密顿量
        """
        raise NotImplementedError

    @abc.abstractmethod
    def clip_parameters(self) -> None:
        """原地裁剪线性和二次偏置权重。"""
        raise NotImplementedError

    def get_ising_matrix(self):
        """将模型转换为伊辛格式。"""
        self.clip_parameters()
        return self._to_ising_matrix()

    @abc.abstractmethod
    def _to_ising_matrix(self):
        """将模型转换为伊辛格式。"""
        raise NotImplementedError("Subclasses must implement _ising method")

    def objective(
            self, s_positive: torch.Tensor, s_negtive: torch.Tensor
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

    def sample(self, sampler: Sampler) -> torch.Tensor:
        """从玻尔兹曼机中采样。

        Args:
            sampler: 用于从模型中采样的优化器。采样器可以是kaiwuSDK的CIM或者其他求解器。

        Returns:
            torch.Tensor: 从模型中采样的自旋
        """
        ising_mat = self.get_ising_matrix()
        solution = sampler.solve(ising_mat)
        # 假设最后一列是能量或其他元数据，我们只取自旋状态
        spins = solution[:, :-1]
        # 将 {-1, 1} 的 Ising 自旋转换为 {0, 1} 的二进制状态
        binary_states = (spins + 1) / 2
        solution = torch.FloatTensor(binary_states)
        solution = solution.to(self.device)
        return solution
