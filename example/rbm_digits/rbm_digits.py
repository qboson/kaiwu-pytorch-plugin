# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""
Restricted Boltzmann Machine
包含RBM的类  以及训练RBM+model/仅训练model的函数 训练RBM+model会保存训练过程中的似然值和预测准确率
"""
import os
import numpy as np
from scipy.ndimage import shift
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import gen_even_slices
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

import torch
from torch.optim import SGD
from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.classical import SimulatedAnnealingOptimizer


def _ensure_result_dir():
    """确保 result 文件夹存在"""
    os.makedirs("results", exist_ok=True)


class RBMRunner(TransformerMixin, BaseEstimator):
    """
    RBMRunner类用于训练和使用受限玻尔兹曼机（RBM）模型。
    Args:
        n_components (int): 隐层单元的数量。
        learning_rate (float): 学习率。
        batch_size (int): 批处理大小。
        n_iter (int): 迭代次数。
        verbose (int): 是否打印训练过程中的信息。
        random_state (int, optional): 随机种子，用于结果的可重复性。
    """

    def __init__(
        self,
        n_components=256,
        *,
        learning_rate=0.1,
        batch_size=100,
        n_iter=30,
        verbose=False,
        plot_img=False,
        random_state=None,
    ):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.plot_img = plot_img
        self.random_state = random_state

        self.sampler = SimulatedAnnealingOptimizer(alpha=0.999, size_limit=100)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rbm = None  # 用于存储训练好的RBM模型

    def gen_digits_image(self, X, size=8):
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

    def fit(self, X, y=None):  # 修改接口以符合scikit-learn约定
        """
        训练RBM模型
        Args:
            X: 训练数据，形状为 (n_samples, n_features)
            y: 忽略，为兼容scikit-learn接口
        """
        # 初始化受限玻尔兹曼机（RBM）模型
        rbm = RestrictedBoltzmannMachine(
            X.shape[1],  # 可见层单元数（特征维度）
            self.n_components,  # 隐层单元数
        )
        rbm.to(self.device)  # 将模型移动到指定设备（CPU/GPU）
        self.rbm = rbm

        # 初始化优化器
        opt_rbm = SGD(rbm.parameters(), lr=self.learning_rate)

        n_samples = X.shape[0]  # 样本数量
        n_batches = int(np.ceil(float(n_samples) / self.batch_size))  # 批次数量
        # 生成每个batch的切片索引
        batch_slices = list(
            gen_even_slices(n_batches * self.batch_size, n_batches, n_samples=n_samples)
        )
        X_torch = torch.FloatTensor(X).to(self.device)  # 转为torch张量并移动到设备
        idx = 0

        # 训练循环
        for iteration in range(1, self.n_iter + 1):
            for step, batch_slice in enumerate(batch_slices):
                idx += 1
                x = X_torch[batch_slice]  # 获取当前batch数据

                x = rbm.get_hidden(x)  # 正相（计算隐层激活）
                # s = rbm.sample(self.sampler)  # 负相（采样重构数据）
                s = rbm.get_visible(x[:, rbm.num_visible :])  # 使用可见层重构
                opt_rbm.zero_grad()  # 梯度清零

                # 计算目标函数（等价于负对数似然），并加权衰减项
                w_weight_decay = 0.02 * torch.sum(rbm.quadratic_coef**2)  # 权重衰减
                b_weight_decay = 0.05 * torch.sum(rbm.linear_bias**2)  # 偏置衰减
                objective = rbm.objective(x, s) + w_weight_decay + b_weight_decay

                # 反向传播并更新参数
                objective.backward()
                opt_rbm.step()

                # 如果verbose，定期评估模型性能和可视化参数
                if self.verbose:
                    print(f"Iteration {idx}, Objective: {objective.item():.6f}")

                    if (idx - 1) % 20 == 0:
                        # 打印权重和偏置的均值与最大值
                        print(
                            f"jmean {torch.abs(rbm.quadratic_coef).mean()}"
                            f" jmax {torch.abs(rbm.quadratic_coef).max()}"
                        )
                        print(
                            f"hmean {torch.abs(rbm.linear_bias).mean()}"
                            f" hmax {torch.abs(rbm.linear_bias).max()}"
                        )

                        if self.plot_img:
                            display_samples = (
                                rbm.sample(self.sampler)
                                .cpu()
                                .numpy()[:20, : rbm.num_visible]
                            )
                            # 生成样本
                            plt.figure(figsize=(16, 2))
                            plt.imshow(self.gen_digits_image(display_samples, 8))
                            plt.title(f"Generated samples at iteration {iteration}")
                            plt.show()
                            _, axes = plt.subplots(1, 2)
                            axes[0].imshow(rbm.quadratic_coef.detach().cpu().numpy())
                            axes[1].imshow(
                                rbm.quadratic_coef.grad.detach().cpu().numpy()
                            )
                            plt.tight_layout()
                            plt.show()

        return self

    def translate_image(self, image, direction):
        "图片转换"
        if direction == "up":
            return shift(image, [-1, 0], mode="constant", cval=0)
        elif direction == "down":
            return shift(image, [1, 0], mode="constant", cval=0)
        elif direction == "left":
            return shift(image, [0, -1], mode="constant", cval=0)
        elif direction == "right":
            return shift(image, [0, 1], mode="constant", cval=0)
        else:
            raise ValueError("Invalid direction. Use 'up', 'down', 'left', or 'right'.")

    def load_data(self, plot_img=False):
        "载入图片数据"
        digits = load_digits()
        images = digits.images  # 8x8 的图像矩阵
        labels = digits.target  # 对应的标签

        # 获取图像数据和标签
        # 扩展数据集
        expanded_images = []
        expanded_labels = []
        for image, label in zip(images, labels):
            # 原始图像
            expanded_images.append(image)
            expanded_labels.append(label)
            # 向四个方向平移
            for direction in ["up", "down", "left", "right"]:
                translated_image = self.translate_image(image, direction)
                expanded_images.append(translated_image)
                expanded_labels.append(label)
                # 将列表转换为 NumPy 数组
        expanded_images = np.array(expanded_images)
        expanded_labels = np.array(expanded_labels)

        # 可视化图像数据和标签
        if plot_img:
            plt.figure(figsize=(16, 9))
            for index in range(5):
                plt.subplot(1, 5, index + 1)
                plt.imshow(expanded_images[index], origin="lower", cmap="gray")
                plt.title("Training: %i\n" % expanded_labels[index], fontsize=18)

        # 将图像数据展平为二维数组 (n_samples, 64)
        n_samples = expanded_images.shape[0]
        data = expanded_images.reshape((n_samples, -1))

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            data, expanded_labels, test_size=0.2, random_state=42
        )

        # 使用sklearn的MinMaxScaler进行归一化
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    # 在RBMRunner类中添加特征提取方法
    def transform(self, X):
        """
        提取隐藏层特征
        Args:
            X: 输入数据，形状为 (n_samples, n_features)
        Returns:
            隐藏层特征，形状为 (n_samples, n_components)
        """
        if self.rbm is None:
            raise ValueError("RBM model not trained yet. Call fit first.")
        X_torch = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            hidden_output = self.rbm.get_hidden(X_torch)
            features = hidden_output[:, self.rbm.num_visible :]
        return features.cpu().numpy()

    # 绘制原始和重构图像
    def plot_images(
        self,
        images,
        labels,
        title="Reconstructed Images",
        save_as="qbm_reconstructed_images",
        save_pdf=False,
    ):
        """
        绘制原始和重构图像
        """
        if self.rbm is None:
            raise ValueError("RBM model not trained yet. Call fit first.")

        # 重构
        with torch.no_grad():
            images = images.to(self.device)
            images_binary = (images > 0.5).float()
            hidden_activations = self.rbm.get_hidden(images_binary)
            reconstructions = self.rbm.sample(self.sampler)[:, : self.rbm.num_visible]

        # 显示原始和重构图像
        num_samples = len(images)

        fig, axes = plt.subplots(
            2,
            num_samples,
            gridspec_kw={"wspace": 0, "hspace": 0.1},
            figsize=(2 * num_samples, 4),
        )

        # 添加文本标签
        if num_samples > 0:
            axes[0, 0].text(
                -2, 3, "original", size=15, verticalalignment="center", rotation=-270
            )
            axes[1, 0].text(
                -2,
                2,
                "reconstructed",
                size=15,
                verticalalignment="center",
                rotation=-270,
            )

        for n in range(num_samples):
            axes[0, n].imshow(images[n].view(8, 8).cpu().numpy(), cmap=plt.cm.gray)
            axes[1, n].imshow(
                reconstructions[n].view(8, 8).cpu().numpy(), cmap=plt.cm.gray
            )
            axes[0, n].set_title(f"Label: {labels[n]}", fontsize=18)
            axes[0, n].axis("off")
            axes[1, n].axis("off")

        # 保存结果
        if save_pdf:
            _ensure_result_dir()
            plt.savefig(
                f"results/{save_as}.pdf", dpi=300, bbox_inches="tight", format="pdf"
            )
        plt.show()

    # 绘制权重
    def plot_weights(self, save_as="qbm_weights", save_pdf=False):
        """绘制权重"""
        weights = self.rbm.quadratic_coef.detach().cpu().numpy()

        fig, axes = plt.subplots(
            8, 16, gridspec_kw={"wspace": 0.1, "hspace": 0.1}, figsize=(16, 7)
        )
        fig.suptitle(f"{self.n_components} components extracted by QBM", fontsize=16)
        fig.subplots_adjust()

        for i, ax in enumerate(axes.flatten()):
            if i < weights.shape[1]:
                ax.imshow(weights[:, i].reshape(8, 8), cmap=plt.cm.gray)
            ax.axis("off")

        # 保存结果
        if save_pdf:
            _ensure_result_dir()
            plt.savefig(
                f"results/{save_as}.pdf", dpi=300, bbox_inches="tight", format="pdf"
            )
        plt.show()

    # 绘制混淆矩阵
    def plot_confusion_matrix(self, y_true, y_pred, title_suffix="", save_pdf=False):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix ({title_suffix})", fontsize=18)
        plt.xlabel("Predicted Label", fontsize=16)
        plt.ylabel("True Label", fontsize=16)
        # plt.xticks(rotation=45)

        # 保存结果
        if save_pdf:
            _ensure_result_dir()
            plt.savefig(
                f"results/rbm_confusion_matrix_{title_suffix}.pdf",
                dpi=300,
                bbox_inches="tight",
                format="pdf",
            )
        plt.tight_layout()
        plt.show()
