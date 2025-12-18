====
概述
====

1. 产品定位
-----------

**Kaiwu-PyTorch-Plugin（KPP）** 是一个专为机器学习设计的 PyTorch 量子计算编程套件，用于玻尔兹曼机（Boltzmann Machine, BM）和受限玻尔兹曼机（Restricted Boltzmann Machine, RBM）的训练与评估。

该插件基于 PyTorch 和 Kaiwu SDK 构建，能够在相干光量子计算机上实现能量神经网络模型的训练和验证，为研究人员和开发者提供易用的接口，快速将能量神经网络应用于各类机器学习任务。

核心价值
^^^^^^^^

- **量子加速**：利用量子计算的并行特性，为玻尔兹曼分布采样提供指数级加速
- **深度学习融合**：结合 PyTorch 灵活定义神经网络的能力，实现自由定义和训练能量神经网络
- **降低门槛**：提供易用的 PyTorch 接口，支持快速实验验证能量神经网络的新思路
- **研究验证**：已发表于 arXiv，高保真特征提取显著提升聚类性能

2. 核心特性
-----------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - 特性
     - 描述
   * - **量子支持**
     - 继承自 Kaiwu SDK，支持调用相干光量子计算机进行玻尔兹曼分布采样
   * - **PyTorch 原生**
     - 与 PyTorch 生态无缝集成，支持 GPU 加速和自动微分
   * - **灵活架构**
     - 支持自定义可见层和隐藏层维度，适应不同规模的任务
   * - **可扩展性**
     - 模块化设计，便于添加新的能量函数或采样方法

3. 产品优势
-----------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - 优势
     - 说明
   * - **原生适配 PyTorch 生态**
     - 直接调用成熟的深度学习工具链，无需学习新的编程范式
   * - **环境要求低**
     - 光量子计算机支持长时间的连续计算，无需复杂的量子纠错
   * - **支持大规模网络**
     - 光量子比特间天然全连接，无需额外拓扑映射，可扩展至大规模网络
   * - **灵活配置**
     - 采样方法和能量函数分离实现，便于添加新的能量函数或采样方法
   * - **示例丰富**
     - 提供多个参考示例，如手写数字识别、Q-VAE 训练等

4. 应用场景
-----------

Kaiwu-PyTorch-Plugin 适用于多种场景，帮助不同用户解决实际问题：

.. list-table::
   :widths: 15 45 40
   :header-rows: 1

   * - 用户类型
     - 实际问题
     - KPP 价值
   * - 研究人员
     - 探索能量神经网络建模方法，验证量子加速算法性能，开展量子机器学习研究
     - 提供量子采样能力和灵活的模型定义，加速研究进程
   * - 开发者
     - 构建基于玻尔兹曼机的应用，开发生成模型，集成量子计算功能到现有系统
     - 模块化设计，支持灵活扩展和快速开发
   * - 企业用户
     - 开发量子机器学习解决方案原型，探索量子计算在 AI 领域的应用
     - 降低开发成本，加速解决方案落地
   * - 学生和教育者
     - 学习玻尔兹曼机和量子计算，开展教学实验和项目
     - 提供直观易用的工具和丰富的示例，辅助教学和学习

5. 核心模块
-----------

Kaiwu-PyTorch-Plugin 采用模块化设计，包含以下核心模块：

.. list-table::
   :widths: 25 40 35
   :header-rows: 1

   * - 模块名称
     - 主要功能
     - 用户价值
   * - **AbstractBoltzmannMachine**
     - 玻尔兹曼机抽象基类，定义通用接口
     - 提供统一的模型接口，便于扩展
   * - **RestrictedBoltzmannMachine**
     - 受限玻尔兹曼机实现，层间全连接、层内无连接
     - 支持高效的并行采样，适用于特征学习
   * - **BoltzmannMachine**
     - 全连接玻尔兹曼机实现
     - 支持更复杂的能量分布建模

6. 典型使用流程
---------------

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
        title = {Kaiwu-PyTorch-Plugin: A PyTorch Plugin for Quantum Boltzmann Machine},
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
