==================================
RBM 分类：手写数字识别（review）
==================================

本教程演示如何使用受限玻尔兹曼机（RBM）在手写数字数据集上进行特征学习与分类。适合初学者理解 RBM 在图像特征提取与分类中的应用流程。

教程目标
--------

完成本教程后，您将学会：

- 使用 RBM 从图像数据中提取特征
- 将 RBM 作为特征提取器与传统分类器结合
- 可视化 RBM 学到的权重和生成的样本
- 评估分类模型的性能

运行环境
--------

**示例位置**: ``example/rbm_digits/rbm_digits.ipynb``

**依赖项**:

.. code-block:: bash

    pip install scikit-learn matplotlib scipy

1. 数据准备
-----------

本教程使用 scikit-learn 内置的 Digits 数据集，包含 8×8 像素的手写数字图像
通过图像平移扩充数据集，提高模型的泛化能力

.. literalinclude:: ../../../../example/rbm_digits/rbm_digits.py
   :pyobject: RBMRunner.load_data


2. 模型训练
-----------

RBMRunner 类中实现RBM的训练过程

.. literalinclude:: ../../../../example/rbm_digits/rbm_digits.py
   :pyobject: RBMRunner.fit

3. 特征提取与分类
-----------------

3.1 训练分类器
^^^^^^^^^^^^^^

使用逻辑回归进行分类：

.. literalinclude:: ../../../../example/rbm_digits/rbm_classifier.py
   :pyobject: train_classifier



3.2 可视化权重
^^^^^^^^^^^^^^

RBM 学到的权重可以理解为"特征检测器"：

.. literalinclude:: ../../../../example/rbm_digits/rbm_digits.py
   :pyobject: RBMRunner.plot_weights


3.2 可视化混淆矩阵
^^^^^^^^^^^^^^^^^^

混淆矩阵展示了分类器在各类别上的表现：

.. literalinclude:: ../../../../example/rbm_digits/rbm_digits.py
   :pyobject: RBMRunner.plot_confusion_matrix

