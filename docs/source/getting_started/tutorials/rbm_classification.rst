==============================
RBM 分类：手写数字识别
==============================

本教程演示如何使用受限玻尔兹曼机（RBM）在手写数字数据集上进行特征学习与分类。适合初学者理解 RBM 在图像特征提取与分类中的应用流程。

教程目标
--------

完成本教程后，您将学会：

- 使用 RBM 从图像数据中提取特征
- 将 RBM 作为特征提取器与传统分类器结合
- 可视化 RBM 学到的权重和生成的样本
- 评估分类模型的性能

运行环境
--------

**示例位置**: ``example/rbm_digits/rbm_digits.ipynb``

**依赖项**:

.. code-block:: bash

    pip install scikit-learn matplotlib scipy

1. 数据准备
-----------

1.1 加载数据集
^^^^^^^^^^^^^^

本教程使用 scikit-learn 内置的 Digits 数据集，包含 8×8 像素的手写数字图像：

.. code-block:: python

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    # 加载数据
    digits = load_digits()
    images = digits.images  # 8x8 的图像矩阵
    labels = digits.target  # 对应的标签（0-9）

    print(f"数据集大小: {images.shape[0]} 张图片")
    print(f"图像尺寸: {images.shape[1]}x{images.shape[2]}")
    print(f"类别数: {len(np.unique(labels))}")

1.2 数据增强
^^^^^^^^^^^^

通过图像平移扩充数据集，提高模型的泛化能力：

.. code-block:: python

    from scipy.ndimage import shift

    def translate_image(image, direction):
        """将图像向指定方向平移1个像素"""
        if direction == "up":
            return shift(image, [-1, 0], mode="constant", cval=0)
        elif direction == "down":
            return shift(image, [1, 0], mode="constant", cval=0)
        elif direction == "left":
            return shift(image, [0, -1], mode="constant", cval=0)
        elif direction == "right":
            return shift(image, [0, 1], mode="constant", cval=0)

    # 扩展数据集
    expanded_images = []
    expanded_labels = []

    for image, label in zip(images, labels):
        # 原始图像
        expanded_images.append(image)
        expanded_labels.append(label)
        # 向四个方向平移
        for direction in ["up", "down", "left", "right"]:
            translated = translate_image(image, direction)
            expanded_images.append(translated)
            expanded_labels.append(label)

    expanded_images = np.array(expanded_images)
    expanded_labels = np.array(expanded_labels)

    print(f"扩充后数据集大小: {expanded_images.shape[0]} 张图片")

1.3 数据预处理
^^^^^^^^^^^^^^

将图像展平并归一化：

.. code-block:: python

    # 展平图像
    n_samples = expanded_images.shape[0]
    data = expanded_images.reshape((n_samples, -1))  # (n_samples, 64)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        data, expanded_labels, test_size=0.2, random_state=42
    )

    # 归一化到 [0, 1]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")

2. 模型训练
-----------

2.1 定义 RBMRunner 类
^^^^^^^^^^^^^^^^^^^^^

封装 RBM 的训练过程：

.. code-block:: python

    import torch
    from torch.optim import SGD
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.utils import gen_even_slices

    from kaiwu.torch_plugin import RestrictedBoltzmannMachine
    from kaiwu.classical import SimulatedAnnealingOptimizer

    class RBMRunner(TransformerMixin, BaseEstimator):
        """RBM 训练器，兼容 scikit-learn 接口"""

        def __init__(
            self,
            n_components=256,
            learning_rate=0.1,
            batch_size=100,
            n_iter=30,
            verbose=False,
        ):
            self.n_components = n_components
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.n_iter = n_iter
            self.verbose = verbose

            self.sampler = SimulatedAnnealingOptimizer(alpha=0.999, size_limit=100)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.rbm = None

        def fit(self, X, y=None):
            """训练 RBM 模型"""
            # 初始化 RBM
            self.rbm = RestrictedBoltzmannMachine(
                X.shape[1],           # 可见层单元数（64）
                self.n_components,    # 隐层单元数
                h_range=[-1, 1],
                j_range=[-1, 1],
            )
            self.rbm.to(self.device)

            # 优化器
            optimizer = SGD(self.rbm.parameters(), lr=self.learning_rate)

            # 准备数据
            n_samples = X.shape[0]
            n_batches = int(np.ceil(float(n_samples) / self.batch_size))
            batch_slices = list(
                gen_even_slices(n_batches * self.batch_size, n_batches, n_samples=n_samples)
            )
            X_torch = torch.FloatTensor(X).to(self.device)

            # 训练循环
            for iteration in range(1, self.n_iter + 1):
                for batch_slice in batch_slices:
                    x = X_torch[batch_slice]

                    # 正相：计算隐层激活
                    h = self.rbm.get_hidden(x)

                    # 负相：重构可见层
                    s = self.rbm.get_visible(h[:, self.rbm.num_visible:])

                    # 计算目标函数（带权重衰减）
                    optimizer.zero_grad()
                    w_decay = 0.02 * torch.sum(self.rbm.quadratic_coef ** 2)
                    b_decay = 0.05 * torch.sum(self.rbm.linear_bias ** 2)
                    objective = self.rbm.objective(h, s) + w_decay + b_decay

                    # 反向传播
                    objective.backward()
                    optimizer.step()

                if self.verbose and iteration % 5 == 0:
                    print(f"Iteration {iteration}/{self.n_iter}, Loss: {objective.item():.4f}")

            return self

        def transform(self, X):
            """提取隐藏层特征"""
            X_torch = torch.FloatTensor(X).to(self.device)
            with torch.no_grad():
                hidden = self.rbm.get_hidden(X_torch)
                features = hidden[:, self.rbm.num_visible:]
            return features.cpu().numpy()

2.2 训练 RBM
^^^^^^^^^^^^

.. code-block:: python

    # 创建并训练 RBM
    rbm_runner = RBMRunner(
        n_components=128,    # 隐层单元数
        learning_rate=0.1,
        batch_size=100,
        n_iter=30,
        verbose=True
    )

    rbm_runner.fit(X_train)
    print("RBM 训练完成！")

3. 特征提取与分类
-----------------

3.1 提取特征
^^^^^^^^^^^^

使用训练好的 RBM 提取特征：

.. code-block:: python

    # 提取训练集和测试集的特征
    X_train_features = rbm_runner.transform(X_train)
    X_test_features = rbm_runner.transform(X_test)

    print(f"原始特征维度: {X_train.shape[1]}")
    print(f"RBM 特征维度: {X_train_features.shape[1]}")

3.2 训练分类器
^^^^^^^^^^^^^^

使用逻辑回归进行分类：

.. code-block:: python

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

    # 训练逻辑回归分类器
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train_features, y_train)

    # 预测
    y_pred = classifier.predict(X_test_features)

    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n分类准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

4. 可视化分析
-------------

4.1 可视化权重
^^^^^^^^^^^^^^

RBM 学到的权重可以理解为"特征检测器"：

.. code-block:: python

    import matplotlib.pyplot as plt

    def plot_weights(rbm, n_components):
        """可视化 RBM 权重"""
        weights = rbm.quadratic_coef.detach().cpu().numpy()

        n_rows = 8
        n_cols = min(16, n_components // n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
        fig.suptitle(f'RBM 学到的 {n_components} 个特征', fontsize=12)

        for i, ax in enumerate(axes.flatten()):
            if i < weights.shape[1]:
                ax.imshow(weights[:, i].reshape(8, 8), cmap='gray')
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    plot_weights(rbm_runner.rbm, rbm_runner.n_components)

4.2 可视化混淆矩阵
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    def plot_confusion_matrix(y_true, y_pred):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.show()

    plot_confusion_matrix(y_test, y_pred)

4.3 生成样本
^^^^^^^^^^^^

使用 RBM 生成新的手写数字图像：

.. code-block:: python

    def generate_samples(rbm, sampler, n_samples=20):
        """从 RBM 生成样本"""
        samples = rbm.sample(sampler)[:n_samples, :rbm.num_visible]
        samples = samples.cpu().numpy()

        # 可视化
        fig, axes = plt.subplots(2, 10, figsize=(15, 3))
        for i, ax in enumerate(axes.flatten()):
            if i < n_samples:
                ax.imshow(samples[i].reshape(8, 8), cmap='gray')
            ax.axis('off')

        plt.suptitle('RBM 生成的样本')
        plt.tight_layout()
        plt.show()

    generate_samples(rbm_runner.rbm, rbm_runner.sampler)

5. 完整代码
-----------

运行完整示例：

.. code-block:: bash

    cd example/rbm_digits
    jupyter notebook rbm_digits.ipynb

或者直接运行 Python 脚本：

.. code-block:: bash

    python rbm_digits.py

6. 下一步
---------

- 尝试调整 ``n_components`` 参数，观察对分类性能的影响
- 学习 :doc:`dbn_classification` 了解如何堆叠多个 RBM 构建深度信念网络
- 探索其他数据集（如 MNIST）上的应用
