========
快速开始
========

本章通过简单的代码示例，帮助您快速上手 Kaiwu-PyTorch-Plugin。

1. 基本示例
-----------

1.1 受限玻尔兹曼机（RBM）
^^^^^^^^^^^^^^^^^^^^^^^^^

以下示例展示了如何使用 ``RestrictedBoltzmannMachine`` 类进行基本的模型训练：

.. code-block:: python

    import torch
    from torch.optim import SGD
    from kaiwu.torch_plugin import RestrictedBoltzmannMachine
    from kaiwu.classical import SimulatedAnnealingOptimizer

    if __name__ == "__main__":
        # 参数设置
        SAMPLE_SIZE = 17
        num_visible = 20
        num_hidden = 30

        # 初始化采样器（使用模拟退火优化器）
        sampler = SimulatedAnnealingOptimizer()

        # 准备输入数据（二值化，取值 -1 或 1）
        x = 1 - 2.0 * torch.randint(0, 2, (SAMPLE_SIZE, num_visible))

        # 实例化 RBM 模型
        rbm = RestrictedBoltzmannMachine(
            num_visible,
            num_hidden,
        )

        # 实例化优化器
        optimizer = SGD(rbm.parameters(), lr=0.01)

        # 训练循环示例（单次迭代）
        # 正相：从数据计算隐藏层表示
        h = rbm.get_hidden(x)

        # 负相：从模型分布采样
        s = rbm.sample(sampler)

        # 梯度清零
        optimizer.zero_grad()

        # 计算目标函数（与负对数似然梯度等价）
        objective = rbm.objective(h, s)

        # 反向传播计算梯度
        objective.backward()

        # 更新模型参数
        optimizer.step()

        print(f"Objective: {objective.item():.4f}")

1.2 玻尔兹曼机（BM）
^^^^^^^^^^^^^^^^^^^^

以下示例展示了如何使用 ``BoltzmannMachine`` 类：

.. code-block:: python

    import torch
    from torch.optim import SGD
    from kaiwu.torch_plugin import BoltzmannMachine
    from kaiwu.classical import SimulatedAnnealingOptimizer

    if __name__ == "__main__":
        # 参数设置
        SAMPLE_SIZE = 17
        num_nodes = 50

        # 初始化采样器
        sampler = SimulatedAnnealingOptimizer()

        # 准备输入数据
        x = 1 - 2.0 * torch.randint(0, 2, (SAMPLE_SIZE, num_nodes))

        # 实例化 BM 模型
        bm = BoltzmannMachine(num_nodes)

        # 实例化优化器
        optimizer = SGD(bm.parameters(), lr=0.01)

        # 训练循环示例
        s = bm.sample(sampler)

        optimizer.zero_grad()
        objective = bm.objective(x, s)
        objective.backward()
        optimizer.step()

        print(f"Objective: {objective.item():.4f}")

2. 完整训练流程
---------------

以下是一个更完整的 RBM 训练示例，包含多个训练轮次和损失记录：

.. code-block:: python

    import torch
    from torch.optim import SGD
    from kaiwu.torch_plugin import RestrictedBoltzmannMachine
    from kaiwu.classical import SimulatedAnnealingOptimizer

    def train_rbm():
        # 超参数
        num_visible = 64      # 可见层节点数（例如 8x8 图像）
        num_hidden = 32       # 隐藏层节点数
        batch_size = 32       # 批大小
        num_epochs = 100      # 训练轮次
        learning_rate = 0.01  # 学习率

        # 初始化模型和优化器
        rbm = RestrictedBoltzmannMachine(num_visible, num_hidden)
        optimizer = SGD(rbm.parameters(), lr=learning_rate)
        sampler = SimulatedAnnealingOptimizer()

        # 模拟训练数据（实际应用中替换为真实数据）
        train_data = torch.randint(0, 2, (500, num_visible)).float() * 2 - 1

        # 训练循环
        losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            # 遍历批次
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]

                # 正相
                h = rbm.get_hidden(batch)

                # 负相
                s = rbm.sample(sampler)

                # 更新参数
                optimizer.zero_grad()
                loss = rbm.objective(h, s)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / (len(train_data) // batch_size)
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        return rbm, losses

    if __name__ == "__main__":
        model, training_losses = train_rbm()
        print("Training completed!")

3. 使用不同的采样器
-------------------

Kaiwu SDK 提供多种采样器，您可以根据需求选择：

.. code-block:: python

    from kaiwu.classical import SimulatedAnnealingOptimizer
    from kaiwu.classical import BruteForceOptimizer

    # 模拟退火优化器（推荐，适用于大多数场景）
    sampler_sa = SimulatedAnnealingOptimizer()

    # 暴力搜索优化器（仅适用于小规模问题）
    sampler_bf = BruteForceOptimizer()

    # 使用量子采样器（需要真机访问权限）
    # from kaiwu.quantum import QuantumSampler
    # sampler_quantum = QuantumSampler()

4. 下一步
---------

恭喜您完成了快速开始！接下来，您可以：

- **学习教程**：查看 :doc:`tutorials/index` 了解更多实际应用案例
- **RBM 分类**：:doc:`tutorials/rbm_classification` - 使用 RBM 进行手写数字分类
- **DBN 分类**：:doc:`tutorials/dbn_classification` - 使用深度信念网络进行分类
- **BM 生成**：:doc:`tutorials/bm_generation` - 使用玻尔兹曼机生成数据
- **Q-VAE**：:doc:`tutorials/qvae_mnist` - 量子变分自编码器生成与表征学习

- **查阅 API 文档**：了解各模块的详细接口和参数
- **探索示例代码**：项目 ``example/`` 目录包含更多完整示例
