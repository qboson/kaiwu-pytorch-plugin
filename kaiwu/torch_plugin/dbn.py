# -*- coding: utf-8 -*-
"""deep belief network DBN模型
包含DBN的类  以及训练DBN+model/仅训练model的函数 训练DBN+model会保存训练过程中的似然值和预测准确率
"""
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from .restricted_boltzmann_machine import RestrictedBoltzmannMachine


# =================== 无监督DBN通用模型 =====================
class UnsupervisedDBN(nn.Module):
    """
    无监督DBN（堆叠RBM）通用模型架构
    参数:
        hidden_layers_structure (list): 每层隐藏单元个数
        device (torch.device): 计算设备
    """

    def __init__(self, hidden_layers_structure=[100, 100]):
        super().__init__()
        self.hidden_layers_structure = hidden_layers_structure

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.rbm_layers = None
        self.input_dim = None
        self._is_trained = False

    def create_rbm_layer(self, input_dim):
        """创建RBM层"""
        self.input_dim = input_dim
        self.rbm_layers = nn.ModuleList()

        current_dim = input_dim
        for n_hidden in self.hidden_layers_structure:
            rbm = RestrictedBoltzmannMachine(
                num_visible=current_dim,  # 可见层单元数（特征维度）
                num_hidden=n_hidden,  # 隐层单元数
                h_range=[-1, 1],  # 隐层偏置范围
                j_range=[-1, 1],  # 权重范围
            ).to(
                self.device
            )  # 将模型移动到指定设备（CPU/GPU）
            self.rbm_layers.append(rbm)
            current_dim = n_hidden

        self._is_trained = False
        return self

    def forward(self, X):
        """前向传播 - 特征变换"""
        if self.rbm_layers is None:
            raise ValueError("Model not built yet. Call create_rbm_layer first.")
        if not self._is_trained:
            raise ValueError(
                "Model not trained yet. Call mark_as_trained() after training."
            )

        X_data = X.astype(np.float32)
        for rbm in self.rbm_layers:
            with torch.no_grad():
                hidden_output = rbm.get_hidden(
                    torch.FloatTensor(X_data).to(self.device)
                )
                X_data = (
                    hidden_output[:, rbm.num_visible :].cpu().numpy()
                )  # 只取隐藏层部分
        return X_data

    def transform(self, X):
        """sklearn兼容的transform方法"""
        return self.forward(X)

    def reconstruct(self, X, layer_index=0):
        """从指定层重建输入"""
        if self.rbm_layers is None or len(self.rbm_layers) == 0:
            raise ValueError("No RBM layers found. Please fit the model first.")

        if layer_index >= len(self.rbm_layers):
            raise ValueError(f"Layer index {layer_index} out of range.")

        rbm = self.rbm_layers[layer_index]
        return self.reconstruct_with_rbm(rbm, X, self.device)

    def mark_as_trained(self):
        """标记模型为已训练状态"""
        self._is_trained = True
        return self

    def get_rbm_layer(self, index):
        """获取指定索引的RBM层"""
        if index < len(self.rbm_layers):
            return self.rbm_layers[index]
        return None

    @staticmethod
    def reconstruct_with_rbm(rbm, X, device=None):
        """
        使用单个RBM重建数据
        """
        if device is None:
            device = rbm.device

            # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).to(device)

        with torch.no_grad():
            # 使用RBM的get_hidden获取隐藏层表示
            hidden_act = rbm.get_hidden(X_tensor)
            hidden_part = hidden_act[:, rbm.num_visible :]  # 只取隐藏层部分

            # 重建可见层（使用权重转置）
            visible_recon = torch.sigmoid(
                torch.matmul(hidden_part, rbm.quadratic_coef.t())
                + rbm.linear_bias[: rbm.num_visible]
            )

            # 计算重建误差
            recon_errors = (
                torch.mean((X_tensor - visible_recon) ** 2, dim=1).cpu().numpy()
            )

        return visible_recon.cpu().numpy(), recon_errors

    @property
    def num_layers(self):
        """返回RBM层数"""
        return len(self.rbm_layers)

    @property
    def output_dim(self):
        """返回输出维度"""
        if len(self.rbm_layers) > 0:
            return self.rbm_layers[-1].num_hidden
        return self.input_dim


# =================== 无监督DBN训练器 =====================
