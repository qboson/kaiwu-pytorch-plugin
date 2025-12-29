==================================
RBM 分类：手写数字识别
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

使用 scikit-learn 内置的 Digits 数据集，包含 8×8 像素的手写数字图像。通过图像平移扩充数据集，提高模型的泛化能力。

此部分教程将介绍如何载入经典的数字手写体数据集，并通过简单的数据增强技术（如向四个方向平移图像）来扩展该数据集，以提高模型的泛化能力。  
首先，我们使用`load_digits`从`sklearn`库中加载数据，获取8x8像素的手写数字图像及其标签。  
接着，对每个原始图像向上下左右四个方向平移，创建更多的训练样本。  
此外，还提供了可视化前几个样本的功能，以便于直观理解数据形态。  
最后，将所有图像数据展平为二维数组，并划分成训练集和测试集，同时进行归一化处理，确保各特征值位于[0, 1]区间内，为后续模型训练做好准备。  

.. literalinclude:: ../../../../example/rbm_digits/rbm_digits.py
   :pyobject: RBMRunner.load_data


2. 模型训练
-----------

此部分讲解了如何训练一个受限玻尔兹曼机（RBM）模型。

首先，基于输入数据的特征维度和预设的隐层单元数初始化RBM模型，并将模型放置在指定设备上（如CPU或GPU）。
接着，定义了随机梯度下降（SGD）优化器用于更新模型参数。整个训练过程采用小批量梯度下降法，通过循环遍历每个批次的数据来逐步调整模型参数以最小化目标函数，该函数包括了负对数似然估计以及权重和偏置的衰减项。

在训练过程中，如果启用了`verbose`模式，则会定期输出当前迭代的目标函数值及模型参数统计信息，并可视化生成的样本图像和模型参数，便于实时监控模型的学习进度和效果。

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

