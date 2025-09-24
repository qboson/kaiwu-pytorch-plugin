# -*- coding: utf-8 -*-
"""deep belief network DBN模型
包含DBN的类  以及训练DBN+model/仅训练model的函数 训练DBN+model会保存训练过程中的似然值和预测准确率
"""
import os
import numpy as np
from scipy.ndimage import shift
import matplotlib.pyplot as plt
import seaborn as sns

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
# from sklearn.utils.multiclass import unique_labels
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, TensorDataset
from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.classical import SimulatedAnnealingOptimizer

# =================== 无监督DBN =====================
class UnsupervisedDBN(BaseEstimator, TransformerMixin):
    """
    基于RestrictedBoltzmannMachine用于无监督预训练的DBN（堆叠RBM）
    参数:
        hidden_layers_structure (list): 每层隐藏单元个数，如[256, 256]表示两层。
        learning_rate_rbm (float): 每层RBM的学习率。
        n_epochs_rbm (int): 每层RBM的训练轮数。
        batch_size (int): 批大小。
        verbose (bool): 打印训练信息。
        shuffle (bool): 数据加载时是否打乱。
        drop_last (bool): DataLoader是否丢弃最后不足batch的样本。
        random_state (int): 随机种子。
    """
    def __init__(
        self, 
        hidden_layers_structure=[100, 100],
        learning_rate_rbm=0.1,
        n_epochs_rbm=10,
        batch_size=100,
        verbose=True,
        shuffle=True,
        drop_last=False,
        plot_img=False,
        random_state=None
        ):
        self.hidden_layers_structure = hidden_layers_structure
        self.learning_rate_rbm = learning_rate_rbm
        self.n_epochs_rbm = n_epochs_rbm
        self.batch_size = batch_size
        self.verbose = verbose
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.plot_img = plot_img
        self.random_state = random_state
        
        self.sampler = SimulatedAnnealingOptimizer(alpha=0.999, size_limit=100)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rbm_layers = None

    def fit(self, X, y=None):  # 修改接口以符合scikit-learn约定
        """
        预训练阶段
        Args:
            X: 训练数据，形状为 (n_samples, n_features)
            y: 忽略，为兼容scikit-learn接口
        """
        # 设置随机种子
        if self.random_state is not None:
            self._set_random_seed()

        input_data = X.astype(np.float32)

        # 清空之前的RBM
        self.rbm_layers = []

        for idx, n_hidden in enumerate(self.hidden_layers_structure):
            n_visible = input_data.shape[1]
            if self.verbose:
                print(f"\n[DBN] Pre-training RBM layer {idx+1}/{len(self.hidden_layers_structure)}: {n_visible} -> {n_hidden}")

            # 创建并训练RBM层
            rbm = self._create_rbm_layer(n_visible, n_hidden)
            input_data = self._train_rbm_layer(rbm, input_data, idx)

        return self

    def transform(self, X):
        """
        特征变换
        Args:
            X: 输入数据，形状为 (n_samples, n_features)
        Returns:
            隐藏层特征，形状为 (n_samples, n_hidden)
        """
        if self.rbm_layers is None:
            raise ValueError("DBN model not trained yet. Call fit first.")
        X_data = X.astype(np.float32)
        for rbm in self.rbm_layers:
            with torch.no_grad():
                hidden_output = rbm.get_hidden(torch.FloatTensor(X_data).to(self.device))
                X_data = hidden_output[:, rbm.num_visible:].cpu().numpy()  # 只取隐藏层部分
        return X_data

    def _set_random_seed(self):
        """设置随机种子"""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)

    def _create_rbm_layer(self, n_visible, n_hidden):
        """创建RBM层"""
        rbm = RestrictedBoltzmannMachine(
            num_visible=n_visible,      # 可见层单元数（特征维度）
            num_hidden=n_hidden,        # 隐层单元数
            h_range=[-1, 1],            # 隐层偏置范围
            j_range=[-1, 1],            # 权重范围
        ).to(self.device)               # 将模型移动到指定设备（CPU/GPU）
        self.rbm_layers.append(rbm)
        return rbm

    def _train_rbm_layer(self, rbm, input_data, layer_idx):
        """训练单个RBM层"""
        optimizer = SGD(rbm.parameters(), lr=self.learning_rate_rbm)

        # 使用当前层的输入数据，而不是原始X
        X_torch = torch.FloatTensor(input_data).to(self.device)
        
        dataset = TensorDataset(X_torch)
        loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle
        )

        if self.verbose:
            print("[DBN] Pre-training start:")

        # 训练循环
        for epoch in range(self.n_epochs_rbm):
            total_loss = 0.0                                      # 当前epoch的总目标值
            for i, (batch_x,) in enumerate(loader):               # 获取batch数据, batch_x: size=[batch, n_visible]
                loss = self._train_batch(rbm, optimizer, batch_x)

                # 累加目标值
                total_loss += loss.item()
                
                # 每隔20个batch打印一次权重和偏置的统计信息
                if self.verbose and i % 20 == 0:
                    self._print_layer_stats(rbm, i, epoch)

                    # 样本和权重可视化
                    if self.plot_img:
                    	self._print_display_samples(rbm, i, epoch)
            
                # 计算当前epoch的平均目标值
                avg_loss = total_loss / len(loader)

                # 每隔5个batch打印一次epoch的平均损失
                if self.verbose and i % 5 == 0:
                    print(f"Iteration {i+1}, Average Loss: {avg_loss:.6f}")

            # 打印每层RBM的平均损失以及数据形状
            if self.verbose:
                print(f"Layer {layer_idx+1}, Epoch {epoch+1}: Loss {avg_loss:.6f}")
                print(f"Output shape after layer {layer_idx+1}: {input_data.shape}")
        
            # 打印每个epoch的平均损失
            if self.verbose:
                print(f"[RBM] Epoch {epoch+1}/{self.n_epochs_rbm} \tAverage Loss: {avg_loss:.6f}\n")
        
        if self.verbose:
            print("[DBN] Pre-training finished")     
        
        # 提取特征作为下一层输入
        with torch.no_grad():
            hidden_output = rbm.get_hidden(X_torch)
            return hidden_output[:, rbm.num_visible:].cpu().numpy()  # 只取隐藏层部分

    def _train_batch(self, rbm, optimizer, batch_x):
        """训练单个batch"""
        h_prob = rbm.get_hidden(batch_x)      # 正相（计算隐层激活）, size=[batch, n_hidden]
        s = rbm.sample(self.sampler)          # 负相（采样重构数据）, size=[[batch, n_visible + n_hidden]
        optimizer.zero_grad()                 # 梯度清零

        # 计算损失函数（等价于负对数似然）+正则项
        w_decay = 0.02 * torch.sum(rbm.quadratic_coef**2)    # 权重衰减
        b_decay = 0.05 * torch.sum(rbm.linear_bias**2)       # 偏置衰减
        loss = rbm.objective(h_prob, s) + w_decay + b_decay
        
        # 反向传播并更新参数
        loss.backward()
        optimizer.step()
        return loss

    def _print_layer_stats(self, rbm, batch_idx, epoch):
        """打印统计信息"""
        # print(f"Batch {batch_idx+1}: \n"
        print(f"jmean {torch.abs(rbm.quadratic_coef).mean().item():.6f} "
              f"jmax {torch.abs(rbm.quadratic_coef).max().item():.6f}")
        print(f"hmean {torch.abs(rbm.linear_bias).mean().item():.6f} "
              f"hmax {torch.abs(rbm.linear_bias).max().item():.6f}")

    def _print_display_samples(self, rbm, batch_idx, epoch):
        """可视化样本与权重"""
        display_samples = (
            rbm.sample(self.sampler)
            .cpu()
            .numpy()[:20, : rbm.num_visible]
        )
        # 生成样本
        plt.figure(figsize=(16, 2))
        plt.imshow(self._gen_digits_image(display_samples, 8))
        plt.title(f"Generated samples at iteration {batch_idx + 1}")
        plt.show()
        _, axes = plt.subplots(1, 2)
        # 权重可视化
        axes[0].imshow(rbm.quadratic_coef.detach().cpu().numpy())
        axes[1].imshow(rbm.quadratic_coef.grad.detach().cpu().numpy())
        plt.tight_layout()
        plt.show()

    def _gen_digits_image(self, X, size=8):
        """
        生成图片
        Args:
            X: 形状为 (20, size * size) 的数组
        Returns：
            拼接后的大图像，形状为 (8, 20 * size)
        """

        plt.rcParams["image.cmap"] = "gray"
        # 先将每个数字的特征向量还原为8x8图像
        digits = X.reshape(20, size, size)  # 形状：(20, 8, 8)
        # 将20个8x8的图片横向拼接
        image = np.hstack(digits)  # 形状：(8, 160)
        return image
