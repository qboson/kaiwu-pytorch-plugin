========
快速开始
========

本章通过简单的代码示例，帮助您快速上手 Kaiwu-PyTorch-Plugin。

1. 基本示例
-----------

1.1 受限玻尔兹曼机（RBM）
^^^^^^^^^^^^^^^^^^^^^^^^^

以下示例展示了如何使用 ``RestrictedBoltzmannMachine`` 类进行基本的模型训练。
可以定义可见节点、隐藏节点的个数，以及自定义二次项和一次项的初始化。

.. literalinclude:: ../../../example/run_rbm.py

1.2 玻尔兹曼机（BM）
^^^^^^^^^^^^^^^^^^^^

以下示例展示了如何使用 ``BoltzmannMachine`` 类：

.. literalinclude:: ../../../example/run_bm.py


3. 使用不同的采样器
-------------------

Kaiwu SDK 提供多种采样器，您可以根据需求选择：

.. code-block:: python

    from kaiwu.classical import SimulatedAnnealingOptimizer

    # Simulated-annealing optimizer (recommended for most scenarios)
    sampler_sa = SimulatedAnnealingOptimizer()

    # To use the quantum sampler (requires real-machine access)
    # from kaiwu.cim import CIMOptimizer

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
