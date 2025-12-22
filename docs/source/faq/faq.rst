========
常见问题
========

本章汇总了使用 Kaiwu-PyTorch-Plugin 时的常见问题和解决方案。

安装相关
--------

Q: 安装时提示 Python 版本不兼容？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A: Kaiwu-PyTorch-Plugin 目前仅支持 **Python 3.10**。请使用 conda 创建对应版本的环境：

.. code-block:: bash

    conda create -n quantum_env python=3.10
    conda activate quantum_env

Q: 无法导入 kaiwu.torch_plugin 模块？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A: 请检查以下几点：

1. 确认已安装 Kaiwu-PyTorch-Plugin：``pip install -e .``
2. 确认当前环境已激活：``conda activate quantum_env``
3. 确认 Kaiwu SDK 已正确安装

如果问题仍然存在，尝试重新安装：

.. code-block:: bash

    pip uninstall kaiwu-torch-plugin
    pip install -e .

Q: Kaiwu SDK 安装失败？
^^^^^^^^^^^^^^^^^^^^^^^

A: Kaiwu SDK 需要单独安装。请：

1. 访问 `Kaiwu SDK 下载页面 <https://platform.qboson.com/sdkDownload>`_
2. 下载对应系统的安装包
3. 按照页面说明进行安装

模型相关（review）
--------

Q: RBM 和 BM 有什么区别？应该使用哪个？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A: 主要区别如下：

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - 特性
     - RBM（受限玻尔兹曼机）
     - BM（全连接玻尔兹曼机）
   * - 连接结构
     - 层内无连接
     - 全连接
   * - 采样效率
     - 较高（可并行）
     - 较低（需顺序）
   * - 推荐场景
     - 大多数场景
     - 需要建模复杂依赖时

**建议**：初学者和大多数应用场景使用 RBM；只有在需要建模复杂全局依赖关系时才考虑使用 BM。

Q: 如何选择隐藏层节点数？
^^^^^^^^^^^^^^^^^^^^^^^^^

A: 隐藏层节点数的选择取决于：

- **数据复杂度**：复杂数据需要更多隐藏节点
- **可见层大小**：通常隐藏层为可见层的 1-2 倍
- **计算资源**：节点越多，计算量越大

**经验法则**：

- 简单数据（如 8x8 Digits）：64-128 个隐藏节点
- 中等数据（如 MNIST）：256-512 个隐藏节点
- 复杂数据：根据实验调整

Q: 训练损失不下降怎么办？
^^^^^^^^^^^^^^^^^^^^^^^^^

A: 尝试以下方法：

1. **调整学习率**：尝试更小（如 0.001）或更大（如 0.1）的学习率
2. **增加权重衰减**：添加 L2 正则化防止过拟合
3. **检查数据预处理**：确保数据已归一化到合适范围
4. **增加训练轮次**：某些情况下需要更长的训练时间

采样相关（review）
--------

Q: 如何获取量子计算机访问权限？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A: 请按以下步骤操作：

1. 在 `QBoson 平台 <https://platform.qboson.com/>`_ 注册账号
2. 通过平台联系官方工作人员
3. 联系邮箱：developer@boseq.com

在获取真机权限前，您可以使用模拟退火优化器进行开发和测试。

Q: 模拟退火采样器参数如何调整？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A: ``SimulatedAnnealingOptimizer`` 的主要参数：

- ``alpha``：退火系数（0.9-0.999），越接近 1 收敛越慢但结果越优
- ``size_limit``：每次采样返回的样本数

.. code-block:: python

    from kaiwu.classical import SimulatedAnnealingOptimizer

    # 快速采样（质量较低）
    fast_sampler = SimulatedAnnealingOptimizer(alpha=0.9, size_limit=50)

    # 高质量采样（速度较慢）
    quality_sampler = SimulatedAnnealingOptimizer(alpha=0.999, size_limit=100)

性能相关（review）
--------

Q: 如何加速训练？
^^^^^^^^^^^^^^^^^

A: 几种加速方法：

1. **使用 GPU**：

   .. code-block:: python

       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       model = model.to(device)

2. **增大批大小**：在 GPU 内存允许的情况下增大 batch_size

3. **减少采样次数**：在训练初期可以减少采样步数

Q: 内存不足怎么办？
^^^^^^^^^^^^^^^^^^^

A: 尝试以下方法：

1. 减小批大小（batch_size）
2. 减少隐藏层节点数
3. 使用梯度累积

.. code-block:: python

    # 梯度累积示例
    accumulation_steps = 4
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        loss = model(batch) / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

其他问题（review）
--------

Q: 如何引用这个项目？
^^^^^^^^^^^^^^^^^^^^^

A: 请使用以下 BibTeX 格式：

.. code-block:: bibtex

    @software{KaiwuPyTorchPlugin,
        title = {Kaiwu-PyTorch-Plugin},
        author = {{QBoson Inc.}},
        year = {2024},
        url = {https://github.com/QBoson/Kaiwu-pytorch-plugin}
    }

Q: 在哪里可以获取更多帮助？
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A: 您可以通过以下渠道获取帮助：

1. **GitHub Issues**：`提交问题 <https://github.com/QBoson/Kaiwu-pytorch-plugin/issues>`_
2. **开发者社区**：玻声量子开发者社区
3. **联系邮箱**：developer@boseq.com
