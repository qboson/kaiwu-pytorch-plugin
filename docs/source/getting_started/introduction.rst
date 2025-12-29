====
概述
====
Kaiwu-PyTorch-Plugin (KPP)
==========================

Kaiwu-PyTorch-Plugin（KPP）是一个 PyTorch 插件，用于在相干光量子计算机上训练和评估玻尔兹曼机（Boltzmann Machine, BM）及其受限形式（Restricted Boltzmann Machine, RBM）。它通过 Kaiwu SDK 调用量子硬件执行玻尔兹曼分布采样，其余计算（如参数更新）在标准 PyTorch 流程中完成。

1. 设计目标
-----------

- 在保持 PyTorch 原生编程体验的前提下，接入光量子采样能力。
- 支持标准 BM/RBM 模型的定义、训练与推理。
- 提供可扩展接口，便于替换采样方法或能量函数。

2. 功能特性
-----------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - 模型支持
     - 标准 RBM 和全连接 BM。
   * - 层配置
     - 可见层与隐藏层维度可自定义。
   * - 采样方式
     - 负相采样通过 Kaiwu SDK 在相干光量子计算机上执行。
   * - PyTorch 集成
     - 模型参数为 ``torch.nn.Parameter``，支持自动微分、GPU 加速，并可与其他 PyTorch 模块组合使用。

3. 扩展机制
-----------

- 能量函数（如 Ising 形式）与采样器解耦。
- 用户可通过实现标准接口替换采样策略（例如切换为经典 MCMC 或其他后端）。
- 支持自定义能量项，适用于非标准 BM 变体。


4. 示例用例
-----------

- 手写数字生成（基于 MNIST 的 RBM 训练）
- Q-VAE（量子变分自编码器）训练流程

5. 适用用户
-----------

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - 研究人员
     - 验证量子采样对能量模型训练的影响。
   * - 开发者
     - 构建混合经典-量子生成模型。
   * - 教育者与学生
     - 教学玻尔兹曼机原理与量子采样实践。

6. 典型使用流程
-----------------------

使用 Kaiwu-PyTorch-Plugin 进行能量神经网络训练的典型流程如下：

1. **数据准备**：加载并预处理训练数据，转换为模型输入格式
2. **模型定义**：实例化 RBM 或 BM 模型，设置可见层和隐藏层维度
3. **优化器配置**：使用 PyTorch 优化器（如 SGD、Adam）管理模型参数
4. **训练循环**：

   - 从训练数据计算隐藏层表示（正相）
   - 使用采样器从模型分布生成样本（负相）
   - 计算目标函数并反向传播梯度
   - 更新模型参数

5. **模型评估**：使用训练好的模型进行特征提取、分类或生成任务

.. code-block:: python

    import torch
    from torch.optim import SGD
    from kaiwu.torch_plugin import RestrictedBoltzmannMachine
    from kaiwu.classical import SimulatedAnnealingOptimizer

    # 1. 准备数据
    x = torch.randint(0, 2, (batch_size, num_visible)).float()

    # 2. 定义模型
    rbm = RestrictedBoltzmannMachine(num_visible, num_hidden)

    # 3. 配置优化器和采样器
    optimizer = SGD(rbm.parameters(), lr=0.01)
    sampler = SimulatedAnnealingOptimizer()

    # 4. 训练循环
    for epoch in range(num_epochs):
        h = rbm.get_hidden(x)           # 正相：计算隐藏层
        s = rbm.sample(sampler)         # 负相：模型采样

        optimizer.zero_grad()
        loss = rbm.objective(h, s)      # 计算目标函数
        loss.backward()                 # 反向传播
        optimizer.step()                # 更新参数

7. 引用方式
-----------

如果 Kaiwu-PyTorch-Plugin 对您的学术研究有帮助，欢迎引用：

.. code-block:: bibtex

    @software{KaiwuPyTorchPlugin,
        title = {Kaiwu-PyTorch-Plugin},
        author = {{QBoson Inc.}},
        year = {2024},
        url = {https://github.com/QBoson/Kaiwu-pytorch-plugin}
    }

相关研究论文：

.. code-block:: bibtex

    @article{QuantumBoostedDeepLearning,
        title = {Quantum-Boosted High-Fidelity Deep Learning},
        author = {{QBoson Research Team}},
        year = {2025},
        url = {https://arxiv.org/pdf/2508.11190}
    }
