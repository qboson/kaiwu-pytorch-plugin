==============================
BM 生成：玻尔兹曼机数据生成（review）
==============================

本教程演示如何使用全连接玻尔兹曼机（Boltzmann Machine, BM）进行无监督训练和数据生成。该方法适合快速生成大量小规模样本的场景。

教程目标
--------

完成本教程后，您将学会：

- 理解全连接玻尔兹曼机与受限玻尔兹曼机的区别
- 使用 KL 散度和对比散度训练 BM
- 实现学习率调度和采样策略
- 可视化生成样本的分布

运行环境
--------

**示例位置**: ``example/bm_generation/``

- ``train_bm.ipynb``: 训练代码
- ``sample_bm.ipynb``: 采样和测试代码

**依赖项**:

.. code-block:: bash

    pip install kaiwu==1.3.0 pandas matplotlib

1. 全连接玻尔兹曼机简介
-----------------------

1.1 BM vs RBM
^^^^^^^^^^^^^

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - 特性
     - 受限玻尔兹曼机（RBM）
     - 全连接玻尔兹曼机（BM）
   * - 连接结构
     - 层间全连接，层内无连接
     - 所有节点全连接
   * - 采样方式
     - 可并行采样
     - 需要顺序采样
   * - 训练效率
     - 较高
     - 较低
   * - 表达能力
     - 受限于双分图结构
     - 更强，可建模任意分布

1.2 适用场景
^^^^^^^^^^^^

全连接玻尔兹曼机适用于：

- 需要建模复杂依赖关系的场景
- 小规模样本生成
- 研究玻尔兹曼分布采样问题

2. 数据准备
-----------

2.1 加载数据
^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    import pandas as pd
    import torch

    class DataLoader:
        """数据加载器"""

        def __init__(self, data_path):
            self.data_path = data_path

        def load(self):
            """加载数据"""
            # 假设数据是 CSV 格式
            df = pd.read_csv(self.data_path)
            data = df.values.astype(np.float32)
            return data

        def preprocess(self, data):
            """预处理：二值化"""
            # 将数据转换为 {-1, 1}
            binary_data = np.sign(data)
            binary_data[binary_data == 0] = 1
            return binary_data

    # 加载示例数据
    # loader = DataLoader('data_input/your_data.csv')
    # data = loader.load()
    # data = loader.preprocess(data)

2.2 生成模拟数据
^^^^^^^^^^^^^^^^

如果没有真实数据，可以生成模拟数据：

.. code-block:: python

    def generate_synthetic_data(n_samples=1000, n_features=20):
        """生成模拟的二值数据"""
        # 生成具有特定模式的数据
        np.random.seed(42)

        # 创建几个聚类中心
        centers = [
            np.array([1] * 10 + [-1] * 10),
            np.array([-1] * 10 + [1] * 10),
            np.array([1, -1] * 10),
        ]

        data = []
        for _ in range(n_samples):
            # 随机选择一个中心
            center = centers[np.random.randint(len(centers))]
            # 添加噪声
            noise = np.random.choice([-1, 1], size=n_features, p=[0.1, 0.9])
            sample = center * noise
            data.append(sample)

        return np.array(data, dtype=np.float32)

    # 生成数据
    data = generate_synthetic_data(n_samples=1000, n_features=20)
    print(f"数据形状: {data.shape}")
    print(f"取值范围: [{data.min()}, {data.max()}]")

3. 模型构建
-----------

3.1 创建 BM 模型
^^^^^^^^^^^^^^^^

.. code-block:: python

    from kaiwu.torch_plugin import BoltzmannMachine
    from kaiwu.classical import SimulatedAnnealingOptimizer

    # 参数设置
    num_visible = 20    # 可见节点数（等于特征维度）
    num_hidden = 10     # 隐藏节点数
    num_nodes = num_visible + num_hidden

    # 创建 BM 模型
    bm = BoltzmannMachine(num_nodes)

    # 查看模型参数
    print(f"总节点数: {num_nodes}")
    print(f"可见节点: {num_visible}")
    print(f"隐藏节点: {num_hidden}")
    print(f"参数数量: {sum(p.numel() for p in bm.parameters())}")

3.2 初始化采样器
^^^^^^^^^^^^^^^^

.. code-block:: python

    # 使用模拟退火优化器作为采样器
    sampler = SimulatedAnnealingOptimizer(
        alpha=0.999,      # 退火系数
        size_limit=100    # 样本数限制
    )

4. 训练流程
-----------

4.1 学习率调度器
^^^^^^^^^^^^^^^^

使用余弦退火学习率调度：

.. code-block:: python

    import math
    from torch.optim.lr_scheduler import LambdaLR

    class CosineScheduleWithWarmup(LambdaLR):
        """带 Warmup 的余弦退火学习率调度器"""

        def __init__(
            self,
            optimizer,
            num_warmup_steps,
            num_training_steps,
            num_cycles=0.5,
            last_epoch=-1,
        ):
            self.num_warmup_steps = num_warmup_steps
            self.num_training_steps = num_training_steps
            self.num_cycles = num_cycles
            super().__init__(optimizer, self.lr_lambda, last_epoch)

        def lr_lambda(self, current_step):
            # Warmup 阶段
            if current_step < self.num_warmup_steps:
                return float(current_step) / float(max(1, self.num_warmup_steps))

            # 余弦退火阶段
            progress = float(current_step - self.num_warmup_steps) / \
                       float(max(1, self.num_training_steps - self.num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * self.num_cycles * 2.0 * progress)))

4.2 训练器实现
^^^^^^^^^^^^^^

.. code-block:: python

    from torch.optim import SGD, Adam

    class BMTrainer:
        """玻尔兹曼机训练器"""

        def __init__(
            self,
            num_visible,
            num_hidden,
            learning_rate=0.01,
            n_epochs=100,
            batch_size=32,
            warmup_steps=10,
        ):
            self.num_visible = num_visible
            self.num_hidden = num_hidden
            self.num_nodes = num_visible + num_hidden
            self.learning_rate = learning_rate
            self.n_epochs = n_epochs
            self.batch_size = batch_size
            self.warmup_steps = warmup_steps

            # 初始化模型
            self.bm = BoltzmannMachine(self.num_nodes)
            self.sampler = SimulatedAnnealingOptimizer(alpha=0.999, size_limit=100)

            # 记录训练历史
            self.history = {'loss': [], 'kl_div': []}

        def train(self, data):
            """训练 BM 模型"""
            # 准备数据
            data_tensor = torch.FloatTensor(data)
            n_samples = data_tensor.shape[0]
            n_batches = n_samples // self.batch_size

            # 优化器
            optimizer = Adam(self.bm.parameters(), lr=self.learning_rate)

            # 学习率调度器
            total_steps = self.n_epochs * n_batches
            scheduler = CosineScheduleWithWarmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps
            )

            # 训练循环
            for epoch in range(self.n_epochs):
                epoch_loss = 0
                indices = torch.randperm(n_samples)

                for batch_idx in range(n_batches):
                    # 获取批次数据
                    start = batch_idx * self.batch_size
                    end = start + self.batch_size
                    batch_indices = indices[start:end]
                    batch_data = data_tensor[batch_indices]

                    # 扩展数据（添加隐藏节点）
                    hidden_init = torch.randint(0, 2, (self.batch_size, self.num_hidden)).float() * 2 - 1
                    full_data = torch.cat([batch_data, hidden_init], dim=1)

                    # 采样
                    samples = self.bm.sample(self.sampler)

                    # 计算目标函数
                    optimizer.zero_grad()
                    loss = self.bm.objective(full_data, samples)

                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / n_batches
                self.history['loss'].append(avg_loss)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.4f}, "
                          f"LR: {scheduler.get_last_lr()[0]:.6f}")

            return self

        def sample(self, n_samples=100):
            """从模型生成样本"""
            samples = []
            for _ in range(n_samples):
                s = self.bm.sample(self.sampler)
                # 只取可见节点
                visible = s[:, :self.num_visible]
                samples.append(visible.detach().numpy())
            return np.concatenate(samples, axis=0)

4.3 执行训练
^^^^^^^^^^^^

.. code-block:: python

    # 创建训练器
    trainer = BMTrainer(
        num_visible=20,
        num_hidden=10,
        learning_rate=0.01,
        n_epochs=100,
        batch_size=32
    )

    # 训练
    trainer.train(data)

    # 绘制损失曲线
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot(trainer.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('BM 训练损失')
    plt.grid(True)
    plt.show()

5. 样本生成与评估
-----------------

5.1 生成样本
^^^^^^^^^^^^

.. code-block:: python

    # 生成新样本
    generated_samples = trainer.sample(n_samples=500)
    print(f"生成样本形状: {generated_samples.shape}")

5.2 分布比较
^^^^^^^^^^^^

.. code-block:: python

    def compare_distributions(original, generated):
        """比较原始数据和生成数据的分布"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 特征均值比较
        ax = axes[0]
        ax.scatter(original.mean(axis=0), generated.mean(axis=0), alpha=0.7)
        ax.plot([-1, 1], [-1, 1], 'r--')
        ax.set_xlabel('原始数据均值')
        ax.set_ylabel('生成数据均值')
        ax.set_title('特征均值比较')

        # 特征方差比较
        ax = axes[1]
        ax.scatter(original.var(axis=0), generated.var(axis=0), alpha=0.7)
        ax.plot([0, 1], [0, 1], 'r--')
        ax.set_xlabel('原始数据方差')
        ax.set_ylabel('生成数据方差')
        ax.set_title('特征方差比较')

        # 样本直方图
        ax = axes[2]
        ax.hist(original.flatten(), bins=3, alpha=0.5, label='原始', density=True)
        ax.hist(generated.flatten(), bins=3, alpha=0.5, label='生成', density=True)
        ax.legend()
        ax.set_title('值分布比较')

        plt.tight_layout()
        plt.show()

    compare_distributions(data, generated_samples)

5.3 条件采样
^^^^^^^^^^^^

给定部分变量的值，采样其余变量：

.. code-block:: python

    def conditional_sample(bm, sampler, visible_data, num_output=5):
        """条件采样：给定输入，采样输出"""
        visible_tensor = torch.FloatTensor(visible_data)

        # 条件采样
        samples = bm.condition_sample(sampler, visible_tensor)

        # 提取输出部分
        output = samples[:, -num_output:].detach().numpy()
        return output

    # 示例：给定前15个特征，采样后5个特征
    input_data = data[:10, :15]
    output_samples = conditional_sample(trainer.bm, trainer.sampler, input_data, num_output=5)
    print(f"条件采样输出: {output_samples.shape}")

6. 保存与加载模型
-----------------

.. code-block:: python

    import pickle

    class ModelSaver:
        """模型保存器"""

        @staticmethod
        def save(model, path):
            """保存模型"""
            state = {
                'state_dict': model.state_dict(),
                'num_nodes': model.num_nodes,
            }
            with open(path, 'wb') as f:
                pickle.dump(state, f)
            print(f"模型已保存到: {path}")

        @staticmethod
        def load(path):
            """加载模型"""
            with open(path, 'rb') as f:
                state = pickle.load(f)
            model = BoltzmannMachine(state['num_nodes'])
            model.load_state_dict(state['state_dict'])
            print(f"模型已从 {path} 加载")
            return model

    # 保存
    ModelSaver.save(trainer.bm, 'bm_model.pkl')

    # 加载
    loaded_bm = ModelSaver.load('bm_model.pkl')

7. 完整代码
-----------

运行完整示例：

.. code-block:: bash

    cd example/bm_generation

    # 训练模型
    jupyter notebook train_bm.ipynb

    # 采样测试
    jupyter notebook sample_bm.ipynb

8. 下一步
---------

- 尝试不同的隐藏节点数量
- 调整学习率和训练轮次
- 学习 :doc:`qvae_mnist` 了解更强大的生成模型 Q-VAE
