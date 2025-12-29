======================================
Q-VAE：量子变分自编码器
======================================

本教程演示如何训练和评估量子变分自编码器（Quantum Variational Autoencoder, Q-VAE）模型。Q-VAE 结合了变分自编码器和量子玻尔兹曼机，能够实现更强大的生成和表征学习能力。

目标
----

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

1、QVAE 原理概括
~~~~~~~~~~~~~~~~

QVAE（Quantum Variational Autoencoder）是一种将 **量子生成模型** 引入 **变分自编码器 (VAE)** 潜空间的生成模型。其核心思想是：

> **用量子玻尔兹曼机（QBM）替代传统 VAE 中的先验分布，从而构建一个具有量子生成能力的潜变量模型。**

模型结构
^^^^^^^^

QVAE 包括以下关键组件：

1. **编码器（Encoder）**  
   将输入数据 :math:`\mathbf{x}` 映射为潜变量的近似后验分布  
   :math:`q_\phi(\mathbf{z}|\mathbf{x})`，通常由神经网络参数化。

2. **先验分布（Prior）**  
   使用 **量子玻尔兹曼机 (QBM)** 建模潜变量 :math:`\mathbf{z}` 的先验分布。哈密顿量为：

   .. math::

      \mathcal{H}_\theta = \sum_l \Gamma_l \sigma_l^x + \sum_l h_l \sigma_l^z + \sum_{l<m} W_{lm} \sigma_l^z \sigma_m^z

3. **解码器（Decoder）**  
   将潜变量 :math:`\mathbf{z}` （或其连续松弛变量 :math:`\boldsymbol{\zeta}` ）映射回数据空间，并使用解码器重建原始数据：

   .. math::

      p_\theta(\mathbf{x} | \boldsymbol{\zeta}) \approx \text{Bernoulli}(f_\theta(\boldsymbol{\zeta}))

训练目标：Q-ELBO
^^^^^^^^^^^^^^^^^^

QVAE 使用一个 **量子下界 (Q-ELBO)** 来近似最大化对数似然：

.. math::

   \mathcal{L}_{\text{Q-ELBO}} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [\log p_\theta(\mathbf{x} | \boldsymbol{\zeta})] - \tilde{H}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}))

QBM 采样与训练
^^^^^^^^^^^^^^

- **正相（positive phase）**：从编码器采样 :math:`\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})`
- **负相（negative phase）**：从 QBM 中采样 :math:`\mathbf{z} \sim p_\theta(\mathbf{z})`，使用 **蒙特卡洛方法** 或 **量子退火器**

把能量作为目标函数，objective 的梯度即为基于正相和负相采样计算的梯度。


2. 模型架构
~~~~~~~~~~~~

2.1 编码器
^^^^^^^^^^

.. literalinclude:: ../../../../example/qvae_mnist/models.py
   :pyobject: Encoder

2.2 解码器
^^^^^^^^^^

.. literalinclude:: ../../../../example/qvae_mnist/models.py
   :pyobject: Decoder

2.3 Q-VAE 完整模型
^^^^^^^^^^^^^^^^^^

参考模块手册中的QVAE类。

3. 数据准备
~~~~~~~~~~~

3.1 加载 MNIST
^^^^^^^^^^^^^^^^^^

.. code:: python

   def setup_data_loaders(root, download=True, batch_size=256, use_cuda=False):
      """
      设置MNIST数据集的数据加载器
      
      Args:
         root (str): 数据存储根目录
         download (bool): 如果数据不存在是否下载，默认为True
         batch_size (int): 每个批次的样本数量，默认为128
         use_cuda (bool): 是否使用GPU，决定是否启用pin_memory优化
      
      Returns:
         tuple: (train_loader, test_loader) 训练和测试数据加载器
      """
      # 数据预处理
      transform = transforms.Compose([
         transforms.ToTensor(),             # 转换为Tensor
         transforms.Lambda(flatten_tensor)  # 展平：将28x28图像展平成784维向量
         # 等效于：x.reshape(-1) 或 x.flatten()
      ])
      
      # 加载训练集
      train_set = datasets.MNIST(
         root=root,           # 数据存储路径
         train=True,          # 加载训练集（共60000个样本）
         transform=transform, # 应用定义的数据变换
         download=download    # 如果数据不存在则自动下载
      )
      
      # 加载测试集
      test_set = datasets.MNIST(
         root=root,           # 数据存储路径  
         train=False,         # 加载测试集（共10000个样本）
         transform=transform  # 应用相同的数据变换
      )

      # 数据加载器配置参数
      # 根据是否使用GPU选择不同的优化参数
      # 将num_workers设为0避免多进程问题
      kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {'num_workers': 0}

      # 创建训练数据加载器
      train_loader = DataLoader(
         dataset=train_set,     # 训练数据集
         batch_size=batch_size, # 每个批次的样本数
         shuffle=True,          # 每个epoch打乱数据顺序，防止模型记忆顺序
         **kwargs               # 解包上述配置参数
      )
      
      # 创建测试数据加载器
      test_loader = DataLoader(
         dataset=test_set,      # 测试数据集
         batch_size=batch_size, # 批次大小（通常与训练集相同）
         shuffle=False,         # 测试集不需要打乱，保证可重复性
         **kwargs               # 解包配置参数
      )

      return train_loader, test_loader

4. 模型训练
~~~~~~~~~~~~~~

.. literalinclude:: ../../../../example/qvae_mnist/train_qvae.py
   :pyobject: train_qvae

5. 可视化与评估
~~~~~~~~~~~~~~~

5.1 训练过程可视化
^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../example/qvae_mnist/visualizers.py
   :pyobject: plot_training_curves

5.3 潜在空间可视化
^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../example/qvae_mnist/visualizers.py
   :pyobject: t_SNE

6. 表征学习与分类
~~~~~~~~~~~~~~~~~~~~~

Q-VAE 学到的表征可用于下游分类任务：

.. literalinclude:: ../../../../example/qvae_mnist/train_qvae.py
   :pyobject: train_mlp_classifier



7. 科研应用：QBM-VAE
~~~~~~~~~~~~~~~~~~~~~~~

Q-VAE 的进阶版本 QBM-VAE 在科研中展示了重要价值：

**单细胞转录组学分析**：

- 显著提升聚类精度
- 检测传统方法无法辨识的新型细胞亚型
- 为靶点发现提供新线索

**相关论文**：`Quantum-Boosted High-Fidelity Deep Learning <https://arxiv.org/pdf/2508.11190>`_