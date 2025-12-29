==================
预备知识
==================

受限玻尔兹曼机（Restricted Boltzmann Machine）是一种基于能量的概率图模型，由可见层（Visible Layer）和隐层（Hidden Layer）组成，层内无连接，层间全连接。
其核心是通过无监督学习学习数据的潜在特征分布。


1. 神经网络基础
========================================

1.1 神经元模型
--------------

人工神经元是神经网络的基本计算单元。给定输入向量 :math:`\mathbf{x} \in \mathbb{R}^n`，其输出为：

.. math::

   a = \phi\left( \mathbf{w}^\top \mathbf{x} + b \right)

其中 :math:`\mathbf{w} \in \mathbb{R}^n` 为权重向量，:math:`b \in \mathbb{R}` 为偏置项，:math:`\phi(\cdot)` 为激活函数。在概率生成模型中，常用 Sigmoid 激活函数：

.. math::

   \sigma(z) = \frac{1}{1 + e^{-z}}

1.2 基于能量的模型
-------------------

与前馈网络不同，能量基模型（Energy-Based Models, EBMs）通过一个标量能量函数 :math:`E(\mathbf{x}; \theta)` 定义数据的概率分布：

.. math::

   P(\mathbf{x}; \theta) = \frac{\exp(-E(\mathbf{x}; \theta))}{Z(\theta)}

其中配分函数（partition function）:

.. math::

   Z(\theta) = \sum_{\mathbf{x}} \exp(-E(\mathbf{x}; \theta))

确保概率归一化。低能量状态对应高概率。



2. 玻尔兹曼机结构
=======================

- 可见层（**v**）：输入数据的显式表示（如像素值）。
- 隐藏层（**h**）：提取的潜在特征。
- 权重矩阵（**w**）：连接可见层与隐层的权重。
- 偏置：可见层偏置（**b**）和隐层偏置（**c**）。

玻尔兹曼机（BM)的拓扑结构是全连接的，而受限玻尔兹曼机通过去掉了可见层和隐藏层内部的链接，
让Gibbs采样的过程更加高效。

由于 RBM 的受限结构，隐变量在给定可见变量时相互独立，其条件概率为：

.. math::

   P(h_j = 1 \mid \mathbf{v}) = \sigma\left( \sum_i w_{ij} v_i + c_j \right)

同理，

.. math::

   P(v_i = 1 \mid \mathbf{h}) = \sigma\left( \sum_j w_{ij} h_j + b_i \right)

3. 能量函数与概率分布
=====================

3.1 能量函数
--------------

RBM 的能量函数定义为：

.. math::

   E(\mathbf{v}, \mathbf{h}) = -\mathbf{v}^T \mathbf{W} \mathbf{h} - \mathbf{b}^T \mathbf{v} - \mathbf{c}^T \mathbf{h}

其中，:math:`\mathbf{v}, \mathbf{h}` 分别是可见层和隐层的状态，:math:`\mathbf{W}` 是连接的权重，:math:`\mathbf{b}, \mathbf{c}` 是一次项系数。

联合概率分布通过玻尔兹曼分布给出：

.. math::

   P(\mathbf{v}, \mathbf{h}) = \frac{e^{-E(\mathbf{v}, \mathbf{h})}}{Z}

其中 :math:`Z` 为配分函数（归一化因子）。可见层的边缘分布为：

.. math::

   P(\mathbf{v}) = \sum_{\mathbf{h}} P(\mathbf{v}, \mathbf{h})

通过最大化似然函数学习参数 :math:`W,b,c` 。目标函数为负对数似然：

.. math::

   \mathcal{L} = -\sum_{\mathbf{v}} \log P(\mathbf{v})

采用对比散度（CD）算法近似梯度，更新规则为：

.. math::

   \Delta W_{ij} = \epsilon \left( \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{recon}} \right)

其中 :math:`\epsilon` 为学习率，:math:`\langle \cdot \rangle_{\text{data}}` 和 :math:`\langle \cdot \rangle_{\text{recon}}` 分别为数据分布和重构分布的期望。

3.2 梯度的推导
-----------------

能量模型的概率可以写成：

.. math::

   p(x; \theta) = \frac{1}{Z} \tilde{p}(x; \theta)

其梯度为：

.. math::

   \nabla_\theta \log p(x; \theta) = \nabla_\theta \log \tilde{p}(x; \theta) - \nabla_\theta \log Z

配分函数的梯度难以直接计算

.. math::

   \nabla_\theta \log Z
   &= \frac{\nabla_\theta Z}{Z} \\
   &= \frac{\nabla_\theta \sum_x \tilde{p}(x)}{Z} \\
   &= \sum_x \frac{\nabla_\theta \tilde{p}(x)}{Z}

对于保证所有的 :math:`x` 都有 :math:`p(x) > 0` 的模型，我们可以用 :math:`\exp(\log \tilde{p}(x))` 代替 :math:`\tilde{p}(x)`。

.. math::

   \frac{\sum_x \nabla_\theta \exp(\log \tilde{p}(x))}{Z}
   &= \frac{\sum_x \exp(\log \tilde{p}(x)) \nabla_\theta \log \tilde{p}(x)}{Z} \\
   &= \frac{\sum_x \tilde{p}(x) \nabla_\theta \log \tilde{p}(x)}{Z} \\
   &= \sum_x p(x) \nabla_\theta \log \tilde{p}(x) \\
   &= \mathbb{E}_{x \sim p(x)} \nabla_\theta \log \tilde{p}(x)

综上，

.. math::
    \nabla_\theta \log p(x; \theta) = \nabla_\theta \log \hat{p}(x; \theta) - \mathbb{E}_{x \sim p(x; \theta)} \nabla_\theta \log \hat{p}(x; \theta)

第二项中 :math:`p(x; \theta)` 实际上是模型预测的 :math:`\mathbf{x}` 的分布，而训练中的第一项是服从实际的数据的分布的。即上式可以写成

.. math::
   \nabla_\theta \log p(x; \theta) = \mathbb{E}_{x \sim p_{\text{data}}} \nabla_\theta \log \hat{p}(x; \theta) - \mathbb{E}_{x \sim p_{\text{model}}} \nabla_\theta \log \hat{p}(x; \theta)

这里我们考虑玻尔兹曼机的能量函数，容易求得

.. math::
   \nabla_W \log \hat{p}(x; W) = v h^\mathrm{T}

只要分别得到 :math:`p_{\text{data}}`, :math:`p_{\text{model}}` 分布下的 :math:`v` 和 :math:`h` 的值即可计算梯度。即为：


.. math::

   \Delta W_{ij} = \epsilon \left( \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{recon}} \right)