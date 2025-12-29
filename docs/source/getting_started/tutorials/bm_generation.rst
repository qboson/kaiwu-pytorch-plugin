======================================
BM 生成：玻尔兹曼机数据生成
======================================

本教程演示如何使用全连接玻尔兹曼机（Boltzmann Machine, BM）进行无监督训练和数据生成。该方法适合快速生成大量小规模样本的场景。


目标
--------

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


1.3 组件
^^^^^^^^^^^^

- **BoltzmannMachine 模型**  
  由 ``kaiwu.torch_plugin.BoltzmannMachine`` 提供，封装了能量函数、采样逻辑和目标函数计算。模型包含可见层与隐层节点，通过可学习的二次项系数（``quadratic_coef``）和线性偏置建模联合分布。

- **采样器（Sampler）**  
  通过传入的 ``worker``（通常为模拟退火或 Gibbs 采样器）执行从模型分布中采样的操作，用于负相采样（``sample``）和正相采样（``condition_sample``）。

- **并行采样框架**  
  利用 Python 多进程（``multiprocessing``）对批量输入数据进行并行正相采样，提升大规模数据生成效率。

1.4 数据生成流程
^^^^^^^^^^^^^^^^^

1. **负相采样**  
   调用 ``self.bm_net.sample(self.worker)`` 从当前 BM 的联合分布中生成一组完整的可见-隐含状态样本 ``state_all``，用于近似模型分布以计算对比散度类目标。

2. **正相采样**  
   对每个输入批次 ``data``，执行两类正相采样：
   
   - **完整条件采样**：以完整输入 ``data`` 为可见层固定值，采样对应的完整状态 ``state_v``。
   - **部分条件采样**：仅固定输入的非输出部分（即 ``data[:, :-num_output]``），对输出维度进行自由采样，得到 ``state_vi``。

   这两类采样分别用于计算：
   
   - **KL 散度项（kl_divergence）**：衡量模型分布与数据分布的差异。
   - **负条件似然项（ncl）**：鼓励模型在给定输入条件下正确重建输出部分。

3. **目标函数构建**  
   最终优化目标为加权组合：

   .. math::

      \mathcal{L} = \alpha \cdot \text{KL\_divergence} + (1 - \alpha) \cdot \text{NCL}

   其中 :math:`\alpha` 由 ``cost_param["alpha"]`` 控制，平衡生成能力与条件一致性。

4. **多进程加速**  
   将一个 batch 的数据按进程数切分，每个子进程独立调用 ``process_solve_graph`` 执行正相采样与概率估计，结果合并后用于梯度计算。

1.5 数据生成特点
^^^^^^^^^^^^^^^^^

- **支持条件生成**：可指定部分可见单元为观测值，其余单元由模型生成，适用于半监督或序列补全任务。
- **可微分训练**：所有采样操作嵌入 PyTorch 计算图，支持端到端反向传播。
- **可视化支持**：训练过程中可实时绘制权重矩阵及其梯度，便于调试与分析。
- **灵活输出结构**：通过 ``num_output`` 参数显式区分“输入”与“输出”可见单元，适配回归/分类等监督设定。


2. 加载数据
-----------
    
.. literalinclude:: ../../../../example/bm_generation/data_loader.py
   :pyobject: CSVDataset

3. 模型构建
-----------

创建 BM 模型，初始化采样器，从而构建训练器

.. literalinclude:: ../../../../example/bm_generation/trainer.py
   :pyobject: Trainer.__init__

4. 训练流程
-----------

4.1 学习率调度器
^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../../example/bm_generation/trainer.py
   :pyobject: CosineScheduleWithWarmup

4.2 训练器实现
^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../example/bm_generation/trainer.py
   :pyobject: Trainer.train


6. 保存与加载模型
------------------

.. literalinclude:: ../../../../example/bm_generation/saver.py
   :pyobject: Saver
