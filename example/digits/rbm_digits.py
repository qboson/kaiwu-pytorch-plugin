"""
Restricted Boltzmann Machine
包含RBM的类  以及训练RBM+model/仅训练model的函数 训练RBM+model会保存训练过程中的似然值和预测准确率
"""

import numpy as np
from scipy.ndimage import shift
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import gen_even_slices
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.optim import SGD
from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.classical import SimulatedAnnealingOptimizer


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
        verbose=0,
        random_state=None,
    ):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state

        self.sampler = SimulatedAnnealingOptimizer(alpha=0.999, size_limit=100)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def fit(self, X, Y_train, X_test, Y_test):
        """
        训练RBM模型
        Args:
            X: 训练数据，形状为 (n_samples, n_features)
            Y_train: 训练标签，形状为 (n_samples,)
            X_test: 测试数据，形状为 (n_samples, n_features)
            Y_test: 测试标签，形状为 (n_samples,)
        """
        rbm = RestrictedBoltzmannMachine(
            X.shape[1],
            self.n_components,
            h_range=[-1, 1],
            j_range=[-1, 1],
        )
        rbm.to(self.device)

        opt_rbm = SGD(rbm.parameters(), lr=self.learning_rate)

        n_samples = X.shape[0]
        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(
            gen_even_slices(n_batches * self.batch_size, n_batches, n_samples=n_samples)
        )
        predic = []
        X_torch = torch.FloatTensor(X).to(self.device)
        idx = 0
        X_test_torch = torch.FloatTensor(X_test).to(self.device)

        for iteration in range(1, self.n_iter + 1):
            for batch_slice in batch_slices:
                idx += 1
                x = X_torch[batch_slice]

                x = rbm.get_hidden(x)  # positive phase
                s = rbm.sample(self.sampler)  # negative phase
                opt_rbm.zero_grad()
                # Compute the objective---this objective yields the same gradient as the negative
                # log likelihood of the model
                w_weight_decay = 0.02 * torch.sum(rbm.quadratic_coef**2)
                b_weight_decay = 0.05 * torch.sum(rbm.linear_bias**2)
                objective = rbm.objective(x, s) + w_weight_decay + b_weight_decay
                # Backpropgate gradients
                objective.backward()
                opt_rbm.step()
                print(f"Iteration {idx}, Objective: {objective.item():.6f}")
                if self.verbose:
                    if (idx - 1) % 5 == 0:
                        rf = linear_model.LogisticRegression(
                            solver="newton-cholesky", tol=1e-5, max_iter=1000, C=500
                        )
                        predic.append(
                            rf.fit(
                                (rbm.get_hidden(X_torch)[:, rbm.num_visible :]), Y_train
                            ).score(
                                (rbm.get_hidden(X_test_torch)[:, rbm.num_visible :]),
                                Y_test,
                            )
                        )
                        print(f"Accuracy: {predic[-1]:.4f}")
                    if (idx - 1) % 20 == 0:
                        print(
                            f"jmean {torch.abs(rbm.quadratic_coef).mean()}"
                            f" jmax {torch.abs(rbm.quadratic_coef).max()}"
                        )
                        print(
                            f"hmean {torch.abs(rbm.linear_bias).mean()}"
                            f" hmax {torch.abs(rbm.linear_bias).max()}"
                        )
                        plt.figure(figsize=(16, 2))
                        display_samples = (
                            rbm.sample(self.sampler)
                            .cpu()
                            .numpy()[:20, : rbm.num_visible]
                        )
                        plt.imshow(self.gen_digits_image(display_samples, 8))
                        plt.title(f"Generated samples at iteration {iteration}")
                        plt.show()
                        _, axes = plt.subplots(1, 2)
                        axes[0].imshow(rbm.quadratic_coef.detach().cpu().numpy())
                        axes[1].imshow(rbm.quadratic_coef.grad.detach().cpu().numpy())
                        plt.tight_layout()
                        plt.show()

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

    def load_data(self):
        "载入图片数据"
        digits = load_digits()
        images = digits.images  # 8x8 的图像矩阵
        labels = digits.target  # 对应的标签
        # 获取图像数据和标签
        plt.imshow(images[2], origin="lower", cmap="gray")

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
