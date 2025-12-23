======================================
Q-VAE：量子变分自编码器（review）
======================================

本教程演示如何训练和评估量子变分自编码器（Quantum Variational Autoencoder, Q-VAE）模型。Q-VAE 结合了变分自编码器和量子玻尔兹曼机，能够实现更强大的生成和表征学习能力。

教程目标
--------

完成本教程后，您将学会：

- 理解 Q-VAE 的架构和工作原理
- 在 MNIST 数据集上训练 Q-VAE
- 进行图像重建和生成
- 使用 Q-VAE 进行表征学习和分类
- 使用 t-SNE 可视化潜在空间

运行环境
--------

**示例位置**: ``example/qvae_mnist/``

- ``train_qvae.ipynb``: 训练 Q-VAE 模型
- ``train_qvae_classifier.ipynb``: 表征学习与分类

**依赖项**:

.. code-block:: bash

    pip install torchvision==0.22.0 torchmetrics[image]

1. Q-VAE 简介
-------------

1.1 变分自编码器（VAE）
^^^^^^^^^^^^^^^^^^^^^^^

VAE 是一种生成模型，由编码器和解码器组成：

::

    输入 x → [编码器] → 潜在变量 z → [解码器] → 重建 x'

传统 VAE 假设潜在变量服从高斯分布。

1.2 Q-VAE 的创新
^^^^^^^^^^^^^^^^

Q-VAE 用量子玻尔兹曼机（QBM）替代高斯先验：

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - 方面
     - 传统 VAE
     - Q-VAE
   * - 先验分布
     - 高斯分布
     - 玻尔兹曼分布
   * - 潜在空间
     - 连续
     - 离散
   * - 采样方式
     - 重参数化技巧
     - 量子采样
   * - 优势
     - 计算简单
     - 更好地建模离散特征

1.3 应用场景
^^^^^^^^^^^^

Q-VAE 特别适合：

- **生物数据**：单细胞转录组学分析
- **化学数据**：分子生成与药物设计
- **材料科学**：材料结构设计
- **异常检测**：工业质量控制

2. 模型架构
-----------

2.1 编码器
^^^^^^^^^^

.. code-block:: python

    import torch
    from torch import nn
    import torch.nn.functional as F

    class Encoder(nn.Module):
        """Q-VAE 编码器"""

        def __init__(self, input_dim, hidden_dim, latent_dim, weight_decay=0.01):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, latent_dim)
            self.weight_decay = weight_decay

        def forward(self, x):
            x = self.fc1(x)
            x = self.norm1(x)
            x = F.tanh(x)
            x = self.fc2(x)
            return x

        def get_weight_decay(self):
            """L2 正则化"""
            return self.weight_decay * (
                torch.sum(self.fc1.weight ** 2) +
                torch.sum(self.fc2.weight ** 2)
            )

2.2 解码器
^^^^^^^^^^

.. code-block:: python

    class Decoder(nn.Module):
        """Q-VAE 解码器"""

        def __init__(self, latent_dim, hidden_dim, output_dim, weight_decay=0.01):
            super().__init__()
            self.fc1 = nn.Linear(latent_dim, hidden_dim)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.weight_decay = weight_decay

        def forward(self, z):
            z = self.fc1(z)
            z = self.norm1(z)
            z = F.tanh(z)
            z = self.fc2(z)
            return z

        def get_weight_decay(self):
            return self.weight_decay * (
                torch.sum(self.fc1.weight ** 2) +
                torch.sum(self.fc2.weight ** 2)
            )

2.3 Q-VAE 完整模型
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from kaiwu.torch_plugin import RestrictedBoltzmannMachine
    from kaiwu.classical import SimulatedAnnealingOptimizer

    class QVAE(nn.Module):
        """量子变分自编码器"""

        def __init__(
            self,
            input_dim=784,      # MNIST: 28x28
            hidden_dim=256,
            latent_dim=64,
            rbm_hidden=32,
        ):
            super().__init__()

            self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
            self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

            # 量子玻尔兹曼机作为先验
            self.rbm = RestrictedBoltzmannMachine(
                num_visible=latent_dim,
                num_hidden=rbm_hidden
            )

            self.sampler = SimulatedAnnealingOptimizer(alpha=0.999, size_limit=100)

        def encode(self, x):
            """编码"""
            return self.encoder(x)

        def decode(self, z):
            """解码"""
            return torch.sigmoid(self.decoder(z))

        def forward(self, x):
            """前向传播"""
            # 编码
            z = self.encode(x)

            # 二值化潜在变量
            z_binary = (z > 0).float() * 2 - 1

            # 解码
            x_recon = self.decode(z_binary)

            return x_recon, z

        def sample(self, n_samples=1):
            """从先验采样生成新样本"""
            # 从 RBM 采样
            z_samples = self.rbm.sample(self.sampler)[:n_samples, :self.rbm.num_visible]

            # 解码
            with torch.no_grad():
                x_gen = self.decode(z_samples)

            return x_gen

3. 数据准备
-----------

3.1 加载 MNIST
^^^^^^^^^^^^^^

.. code-block:: python

    import torchvision
    from torchvision import transforms
    from torch.utils.data import DataLoader

    # 数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # 展平为 784 维
    ])

    # 加载数据集
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transform,
        download=True
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")

4. 模型训练
-----------

4.1 损失函数
^^^^^^^^^^^^

Q-VAE 的损失函数包含：

- **重建损失**：衡量重建质量
- **KL 散度**：正则化潜在空间
- **权重衰减**：防止过拟合

.. code-block:: python

    def compute_loss(model, x, x_recon, z):
        """计算 Q-VAE 损失"""
        # 重建损失（二元交叉熵）
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

        # KL 散度（使用 RBM 目标函数近似）
        h = model.rbm.get_hidden(z)
        s = model.rbm.sample(model.sampler)
        kl_loss = model.rbm.objective(h, s)

        # 权重衰减
        weight_decay = model.encoder.get_weight_decay() + model.decoder.get_weight_decay()

        # 总损失
        total_loss = recon_loss + kl_loss + weight_decay

        return total_loss, recon_loss, kl_loss

4.2 训练循环
^^^^^^^^^^^^

.. code-block:: python

    from torch.optim import Adam

    def train_qvae(model, train_loader, n_epochs=50, lr=1e-3, device='cuda'):
        """训练 Q-VAE"""
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=lr)

        history = {'loss': [], 'recon': [], 'kl': []}

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0
            epoch_recon = 0
            epoch_kl = 0

            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(device)

                # 前向传播
                x_recon, z = model(data)

                # 计算损失
                loss, recon, kl = compute_loss(model, data, x_recon, z)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_recon += recon.item()
                epoch_kl += kl.item()

            # 记录历史
            n_batches = len(train_loader)
            history['loss'].append(epoch_loss / n_batches)
            history['recon'].append(epoch_recon / n_batches)
            history['kl'].append(epoch_kl / n_batches)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, "
                      f"Loss: {history['loss'][-1]:.2f}, "
                      f"Recon: {history['recon'][-1]:.2f}, "
                      f"KL: {history['kl'][-1]:.4f}")

        return model, history

    # 训练
    model = QVAE()
    model, history = train_qvae(model, train_loader, n_epochs=50)

5. 可视化与评估
---------------

5.1 重建效果
^^^^^^^^^^^^

.. code-block:: python

    import matplotlib.pyplot as plt

    def visualize_reconstruction(model, test_loader, n_samples=10, device='cuda'):
        """可视化重建效果"""
        model.eval()

        # 获取测试样本
        data, labels = next(iter(test_loader))
        data = data[:n_samples].to(device)

        # 重建
        with torch.no_grad():
            recon, _ = model(data)

        # 可视化
        fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))

        for i in range(n_samples):
            # 原始图像
            axes[0, i].imshow(data[i].cpu().view(28, 28), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('原始', fontsize=12)

            # 重建图像
            axes[1, i].imshow(recon[i].cpu().view(28, 28), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel('重建', fontsize=12)

        plt.suptitle('Q-VAE 重建效果')
        plt.tight_layout()
        plt.show()

    visualize_reconstruction(model, test_loader)

5.2 生成新样本
^^^^^^^^^^^^^^

.. code-block:: python

    def visualize_generation(model, n_samples=20, device='cuda'):
        """可视化生成样本"""
        model.eval()

        # 生成样本
        with torch.no_grad():
            generated = model.sample(n_samples).cpu()

        # 可视化
        n_cols = 10
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
        axes = axes.flatten()

        for i in range(n_samples):
            axes[i].imshow(generated[i].view(28, 28), cmap='gray')
            axes[i].axis('off')

        # 隐藏多余的子图
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')

        plt.suptitle('Q-VAE 生成的样本')
        plt.tight_layout()
        plt.show()

    visualize_generation(model, n_samples=20)

5.3 潜在空间可视化
^^^^^^^^^^^^^^^^^^

使用 t-SNE 可视化潜在空间的结构：

.. code-block:: python

    from sklearn.manifold import TSNE

    def visualize_latent_space(model, test_loader, device='cuda'):
        """使用 t-SNE 可视化潜在空间"""
        model.eval()

        latent_vectors = []
        labels_list = []

        # 收集潜在向量
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                z = model.encode(data)
                latent_vectors.append(z.cpu().numpy())
                labels_list.append(labels.numpy())

        latent_vectors = np.concatenate(latent_vectors, axis=0)
        labels_list = np.concatenate(labels_list, axis=0)

        # t-SNE 降维
        print("正在进行 t-SNE 降维...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        latent_2d = tsne.fit_transform(latent_vectors[:5000])  # 使用部分数据加速
        labels_subset = labels_list[:5000]

        # 可视化
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            latent_2d[:, 0],
            latent_2d[:, 1],
            c=labels_subset,
            cmap='tab10',
            alpha=0.6,
            s=5
        )
        plt.colorbar(scatter, label='数字类别')
        plt.title('Q-VAE 潜在空间 (t-SNE)')
        plt.xlabel('t-SNE 维度 1')
        plt.ylabel('t-SNE 维度 2')
        plt.show()

    visualize_latent_space(model, test_loader)

6. 表征学习与分类
-----------------

Q-VAE 学到的表征可用于下游分类任务：

.. code-block:: python

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

    def evaluate_representation(model, train_loader, test_loader, device='cuda'):
        """评估 Q-VAE 表征的分类性能"""
        model.eval()

        def extract_features(loader):
            features, labels = [], []
            with torch.no_grad():
                for data, label in loader:
                    data = data.to(device)
                    z = model.encode(data)
                    features.append(z.cpu().numpy())
                    labels.append(label.numpy())
            return np.concatenate(features), np.concatenate(labels)

        # 提取特征
        X_train, y_train = extract_features(train_loader)
        X_test, y_test = extract_features(test_loader)

        print(f"训练集特征: {X_train.shape}")
        print(f"测试集特征: {X_test.shape}")

        # 训练分类器
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(X_train, y_train)

        # 评估
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n分类准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))

        return accuracy

    accuracy = evaluate_representation(model, train_loader, test_loader)

7. 保存与加载模型
-----------------

.. code-block:: python

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
    }, 'qvae_model.pth')

    # 加载模型
    checkpoint = torch.load('qvae_model.pth')
    model = QVAE()
    model.load_state_dict(checkpoint['model_state_dict'])
    print("模型加载成功！")

8. 完整代码
-----------

运行完整示例：

.. code-block:: bash

    cd example/qvae_mnist

    # 训练 Q-VAE
    jupyter notebook train_qvae.ipynb

    # 表征学习与分类
    jupyter notebook train_qvae_classifier.ipynb

9. 科研应用：QBM-VAE
--------------------

Q-VAE 的进阶版本 QBM-VAE 在科研中展示了重要价值：

**单细胞转录组学分析**：

- 显著提升聚类精度
- 检测传统方法无法辨识的新型细胞亚型
- 为靶点发现提供新线索

**相关论文**：`Quantum-Boosted High-Fidelity Deep Learning <https://arxiv.org/pdf/2508.11190>`_

10. 下一步
----------

- 尝试不同的潜在空间维度
- 在其他数据集（如 Fashion-MNIST）上实验
- 探索 QBM-VAE 在生物信息学中的应用
- 查阅 API 文档了解更多高级功能
