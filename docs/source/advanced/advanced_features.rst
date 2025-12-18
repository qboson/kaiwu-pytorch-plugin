========
高级功能
========

本章介绍 Kaiwu-PyTorch-Plugin 的高级功能，帮助您充分利用量子计算的能力。

1. 自定义采样器
---------------

1.1 采样器接口
^^^^^^^^^^^^^^

Kaiwu SDK 提供了统一的采样器接口，您可以根据需求选择或自定义采样器：

.. code-block:: python

    from kaiwu.classical import SimulatedAnnealingOptimizer
    from kaiwu.classical import BruteForceOptimizer

    # 模拟退火采样器（推荐）
    sa_sampler = SimulatedAnnealingOptimizer(
        alpha=0.999,      # 退火系数，越接近1收敛越慢但结果更优
        size_limit=100    # 每次采样返回的样本数
    )

    # 暴力搜索采样器（仅适用于小规模问题）
    bf_sampler = BruteForceOptimizer()

1.2 量子采样器
^^^^^^^^^^^^^^

使用真正的量子计算机进行采样：

.. code-block:: python

    # 需要配置 Kaiwu SDK 授权
    from kaiwu.quantum import QuantumSampler

    # 创建量子采样器
    quantum_sampler = QuantumSampler(
        device_id="your_device_id",
        token="your_token"
    )

    # 在 RBM 中使用量子采样器
    samples = rbm.sample(quantum_sampler)

2. 自定义能量函数
-----------------

2.1 能量函数基础
^^^^^^^^^^^^^^^^

玻尔兹曼机的能量函数定义了模型的概率分布：

.. math::

    E(\mathbf{v}, \mathbf{h}) = -\sum_i a_i v_i - \sum_j b_j h_j - \sum_{i,j} v_i W_{ij} h_j

其中 :math:`\mathbf{v}` 是可见层，:math:`\mathbf{h}` 是隐藏层。

2.2 扩展能量函数
^^^^^^^^^^^^^^^^

您可以通过继承基类来自定义能量函数：

.. code-block:: python

    from kaiwu.torch_plugin import AbstractBoltzmannMachine
    import torch

    class CustomBoltzmannMachine(AbstractBoltzmannMachine):
        """自定义玻尔兹曼机"""

        def __init__(self, num_visible, num_hidden):
            super().__init__(num_visible + num_hidden)
            self.num_visible = num_visible
            self.num_hidden = num_hidden

            # 自定义参数
            self.custom_param = torch.nn.Parameter(torch.randn(num_visible))

        def energy(self, state):
            """自定义能量函数"""
            v = state[:, :self.num_visible]
            h = state[:, self.num_visible:]

            # 基础能量
            base_energy = super().energy(state)

            # 添加自定义项
            custom_term = torch.sum(self.custom_param * v, dim=1)

            return base_energy + custom_term

3. 高级训练策略
---------------

3.1 学习率调度
^^^^^^^^^^^^^^

使用学习率调度提高训练效果：

.. code-block:: python

    from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 余弦退火
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

    # 或阶梯衰减
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer)
        scheduler.step()

3.2 权重衰减与正则化
^^^^^^^^^^^^^^^^^^^^

防止过拟合的常用技术：

.. code-block:: python

    # L2 正则化
    weight_decay = 0.02 * torch.sum(rbm.quadratic_coef ** 2)
    bias_decay = 0.05 * torch.sum(rbm.linear_bias ** 2)

    total_loss = objective + weight_decay + bias_decay

3.3 对比散度变体
^^^^^^^^^^^^^^^^

对比散度（CD）算法有多种变体：

- **CD-1**：单步吉布斯采样，计算快但近似粗糙
- **CD-k**：k 步吉布斯采样，k 越大近似越精确
- **PCD**：持久对比散度，使用持久马尔可夫链

.. code-block:: python

    def contrastive_divergence(rbm, data, k=1):
        """k 步对比散度"""
        v = data
        for _ in range(k):
            h = rbm.sample_hidden(v)
            v = rbm.sample_visible(h)
        return v

4. 模型组合
-----------

4.1 DBN 与分类器结合
^^^^^^^^^^^^^^^^^^^^

将 DBN 作为特征提取器与下游任务结合：

.. code-block:: python

    from kaiwu.torch_plugin.dbn import UnsupervisedDBN
    from sklearn.ensemble import RandomForestClassifier

    # 预训练 DBN
    dbn = UnsupervisedDBN([256, 128, 64])
    dbn = trainer.train(dbn, X_train)

    # 提取特征
    features = dbn.transform(X_train)

    # 使用任意分类器
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(features, y_train)

4.2 混合架构
^^^^^^^^^^^^

将量子玻尔兹曼机与传统神经网络结合：

.. code-block:: python

    class HybridModel(nn.Module):
        """混合量子-经典模型"""

        def __init__(self):
            super().__init__()
            # 经典编码器
            self.encoder = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 64)
            )
            # 量子层
            self.rbm = RestrictedBoltzmannMachine(64, 32)
            # 经典解码器
            self.decoder = nn.Sequential(
                nn.Linear(32, 256),
                nn.ReLU(),
                nn.Linear(256, 784)
            )

        def forward(self, x):
            z = self.encoder(x)
            # 通过 RBM 处理
            h = self.rbm.get_hidden(z)
            h = h[:, self.rbm.num_visible:]
            out = self.decoder(h)
            return out

5. 性能优化
-----------

5.1 GPU 加速
^^^^^^^^^^^^

利用 GPU 加速训练：

.. code-block:: python

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 移动模型到 GPU
    model = model.to(device)

    # 数据也需要移动到 GPU
    data = data.to(device)

5.2 批处理优化
^^^^^^^^^^^^^^

合理设置批大小：

.. code-block:: python

    # 较大的批大小提高 GPU 利用率
    batch_size = 256 if torch.cuda.is_available() else 64

    # 使用 DataLoader 的 pin_memory 加速数据传输
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

5.3 混合精度训练
^^^^^^^^^^^^^^^^

使用混合精度加速训练：

.. code-block:: python

    from torch.cuda.amp import autocast, GradScaler

    scaler = GradScaler()

    for data in loader:
        optimizer.zero_grad()

        with autocast():
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

6. 调试与可视化
---------------

6.1 权重可视化
^^^^^^^^^^^^^^

.. code-block:: python

    def visualize_weights(rbm, save_path=None):
        """可视化 RBM 权重矩阵"""
        import matplotlib.pyplot as plt

        weights = rbm.quadratic_coef.detach().cpu().numpy()

        plt.figure(figsize=(10, 8))
        plt.imshow(weights, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Weight Value')
        plt.xlabel('Hidden Units')
        plt.ylabel('Visible Units')
        plt.title('RBM Weight Matrix')

        if save_path:
            plt.savefig(save_path)
        plt.show()

6.2 训练监控
^^^^^^^^^^^^

.. code-block:: python

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter('runs/rbm_training')

    for epoch in range(num_epochs):
        loss = train_one_epoch(model, loader)

        # 记录损失
        writer.add_scalar('Loss/train', loss, epoch)

        # 记录权重分布
        writer.add_histogram('Weights', model.quadratic_coef, epoch)

    writer.close()
