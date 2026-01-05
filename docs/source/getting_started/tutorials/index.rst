==================
新手教程
==================

本章提供详细的实践教程，帮助您深入了解 Kaiwu-PyTorch-Plugin 的各种应用场景。

教程概览
--------

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - 教程名称
     - 任务类型
     - 描述
   * - :doc:`rbm_classification`
     - 分类
     - 使用受限玻尔兹曼机（RBM）在手写数字数据集上进行特征学习与分类
   * - :doc:`dbn_classification`
     - 分类
     - 使用深度信念网络（DBN）构建层次化特征表示并进行分类
   * - :doc:`bm_generation`
     - 生成
     - 使用全连接玻尔兹曼机（BM）进行数据生成
   * - :doc:`qvae_mnist`
     - 生成/表征
     - 使用量子变分自编码器（Q-VAE）进行图像生成与表征学习

推荐学习顺序
------------

**初学者路径**：

1. 先完成 :doc:`../quickstart` 了解基本 API
2. 学习 :doc:`rbm_classification` 理解 RBM 的应用流程

**生成模型路径**：

1. 学习 :doc:`bm_generation` 了解玻尔兹曼机的生成能力
2. 进阶到 :doc:`qvae_mnist` 学习更强大的生成模型

**完整学习路径**：

按顺序完成所有教程，全面掌握 Kaiwu-PyTorch-Plugin 的功能。

.. toctree::
   :maxdepth: 2
   :hidden:

   rbm_classification
   dbn_classification
   bm_generation
   qvae_mnist
