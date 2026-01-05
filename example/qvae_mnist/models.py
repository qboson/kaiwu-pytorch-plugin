# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
import torch.nn.functional as F


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


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], output_dim=10):
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
