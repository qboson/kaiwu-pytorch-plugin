==============================
DBN 分类：深度信念网络（review）
==============================

本教程在 RBM 分类的基础上，进一步构建深度信念网络（Deep Belief Network, DBN），通过堆叠多层 RBM 实现更强大的特征学习能力。

教程目标
--------

完成本教程后，您将学会：

- 理解深度信念网络的层次化特征学习
- 实现逐层贪婪的无监督预训练
- 使用两种训练策略：微调模式和分类器模式
- 比较 DBN 与单层 RBM 的性能差异

运行环境
--------

**示例位置**: ``example/dbn_digits/supervised_dbn_digits.ipynb``

**依赖项**:

.. code-block:: bash

    pip install scikit-learn matplotlib scipy

1. 深度信念网络简介
-------------------

1.1 什么是 DBN
^^^^^^^^^^^^^^

深度信念网络是由多层受限玻尔兹曼机堆叠而成的深度生成模型：

::

    输入层 (可见层)
        ↓
    RBM 第1层 → 隐藏层1
        ↓
    RBM 第2层 → 隐藏层2
        ↓
       ...
        ↓
    RBM 第N层 → 输出层

1.2 DBN 的优势
^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - 特性
     - 说明
   * - **层次化特征抽象**
     - 每层学习越来越抽象的特征表示
   * - **无监督预训练**
     - 可以利用大量无标签数据
   * - **避免梯度消失**
     - 逐层训练避免了深度网络的梯度问题
   * - **灵活的训练策略**
     - 支持微调和分类器两种模式

2. 数据准备
-----------

与 RBM 教程相同，使用 Digits 数据集：

.. code-block:: python

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from scipy.ndimage import shift
    import numpy as np

    def translate_image(image, direction):
        """图像平移"""
        shifts = {"up": [-1, 0], "down": [1, 0], "left": [0, -1], "right": [0, 1]}
        return shift(image, shifts[direction], mode="constant", cval=0)

    def load_and_augment_data():
        """加载并增强数据"""
        digits = load_digits()
        images, labels = digits.images, digits.target

        # 数据增强
        expanded_images, expanded_labels = [], []
        for image, label in zip(images, labels):
            expanded_images.append(image)
            expanded_labels.append(label)
            for direction in ["up", "down", "left", "right"]:
                expanded_images.append(translate_image(image, direction))
                expanded_labels.append(label)

        data = np.array(expanded_images).reshape(len(expanded_images), -1)
        labels = np.array(expanded_labels)

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, random_state=42
        )

        # 归一化
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = load_and_augment_data()
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

3. 构建 DBN 模型
----------------

3.1 定义 DBN 架构
^^^^^^^^^^^^^^^^^

.. code-block:: python

    from kaiwu.torch_plugin.dbn import UnsupervisedDBN

    # 定义 DBN 架构
    # 输入: 64 → 隐藏层1: 128 → 隐藏层2: 64 → 隐藏层3: 32
    hidden_layers = [128, 64, 32]

    # 创建 DBN
    dbn = UnsupervisedDBN(hidden_layers)
    print(f"DBN 架构: 64 → {' → '.join(map(str, hidden_layers))}")

3.2 DBN 训练器
^^^^^^^^^^^^^^

使用 ``DBNTrainer`` 进行逐层预训练：

.. code-block:: python

    import torch
    from torch.optim import SGD
    from torch.utils.data import DataLoader, TensorDataset
    from kaiwu.classical import SimulatedAnnealingOptimizer

    class DBNTrainer:
        """DBN 逐层预训练器"""

        def __init__(
            self,
            learning_rate=0.1,
            n_epochs=10,
            batch_size=100,
            verbose=True,
        ):
            self.learning_rate = learning_rate
            self.n_epochs = n_epochs
            self.batch_size = batch_size
            self.verbose = verbose
            self.sampler = SimulatedAnnealingOptimizer(alpha=0.999, size_limit=100)

        def train(self, dbn, data):
            """逐层预训练 DBN"""
            input_data = data.astype(np.float32)

            # 创建第一层 RBM
            if dbn.num_layers == 0:
                dbn.create_rbm_layer(data.shape[1])

            # 逐层训练
            for idx in range(dbn.num_layers):
                rbm = dbn.get_rbm_layer(idx)

                if self.verbose:
                    print(f"\n[DBN] 预训练第 {idx+1}/{dbn.num_layers} 层: "
                          f"{rbm.num_visible} → {rbm.num_hidden}")

                # 训练当前层
                input_data = self._train_layer(rbm, input_data, idx)

            dbn.mark_as_trained()
            return dbn

        def _train_layer(self, rbm, data, layer_idx):
            """训练单层 RBM"""
            optimizer = SGD(rbm.parameters(), lr=self.learning_rate)
            data_tensor = torch.FloatTensor(data).to(rbm.device)

            dataset = TensorDataset(data_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            for epoch in range(self.n_epochs):
                epoch_loss = 0
                for batch in loader:
                    x = batch[0]

                    # 正相
                    h = rbm.get_hidden(x)
                    # 负相
                    s = rbm.get_visible(h[:, rbm.num_visible:])

                    # 更新参数
                    optimizer.zero_grad()
                    loss = rbm.objective(h, s)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                if self.verbose and (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{self.n_epochs}, Loss: {epoch_loss/len(loader):.4f}")

            # 返回隐藏层输出作为下一层输入
            with torch.no_grad():
                hidden = rbm.get_hidden(data_tensor)
                output = hidden[:, rbm.num_visible:].cpu().numpy()

            return output

4. 训练策略
-----------

DBN 支持两种训练策略：

4.1 策略一：微调模式（Fine-tuning）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

预训练后，对整个网络进行端到端的反向传播微调：

.. code-block:: python

    from kaiwu.torch_plugin.dbn import SupervisedDBNClassification

    # 创建监督式 DBN 分类器
    dbn_classifier = SupervisedDBNClassification(
        hidden_layers_structure=[128, 64, 32],
        n_classes=10,
        learning_rate=0.1,
        n_epochs_rbm=10,      # 预训练轮次
        n_epochs_finetune=50, # 微调轮次
        batch_size=100,
    )

    # 训练（自动完成预训练 + 微调）
    dbn_classifier.fit(X_train, y_train)

    # 预测
    y_pred = dbn_classifier.predict(X_test)

    # 评估
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"微调模式准确率: {accuracy:.4f}")

4.2 策略二：分类器模式（Classifier）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 DBN 提取的特征上使用传统分类器：

.. code-block:: python

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    # 训练 DBN（无监督预训练）
    trainer = DBNTrainer(
        learning_rate=0.1,
        n_epochs=10,
        batch_size=100,
        verbose=True
    )

    dbn = UnsupervisedDBN([128, 64, 32])
    dbn = trainer.train(dbn, X_train)

    # 提取特征
    X_train_features = dbn.transform(X_train)
    X_test_features = dbn.transform(X_test)

    print(f"原始维度: {X_train.shape[1]} → DBN 特征维度: {X_train_features.shape[1]}")

    # 使用逻辑回归
    lr_classifier = LogisticRegression(max_iter=1000)
    lr_classifier.fit(X_train_features, y_train)
    y_pred_lr = lr_classifier.predict(X_test_features)
    print(f"逻辑回归准确率: {accuracy_score(y_test, y_pred_lr):.4f}")

    # 使用 SVM
    svm_classifier = SVC(kernel='rbf')
    svm_classifier.fit(X_train_features, y_train)
    y_pred_svm = svm_classifier.predict(X_test_features)
    print(f"SVM 准确率: {accuracy_score(y_test, y_pred_svm):.4f}")

5. 与 RBM 的比较
----------------

比较单层 RBM 和多层 DBN 的性能：

.. code-block:: python

    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt

    def compare_models(X_train, X_test, y_train, y_test):
        """比较 RBM 和 DBN 的性能"""
        results = {}

        # 单层 RBM (128 hidden units)
        from kaiwu.torch_plugin import RestrictedBoltzmannMachine
        rbm = RestrictedBoltzmannMachine(64, 128)
        # ... 训练 RBM 并提取特征
        # results['RBM-128'] = accuracy

        # 两层 DBN (128 → 64)
        dbn_2layer = UnsupervisedDBN([128, 64])
        # ... 训练并评估
        # results['DBN-2层'] = accuracy

        # 三层 DBN (128 → 64 → 32)
        dbn_3layer = UnsupervisedDBN([128, 64, 32])
        # ... 训练并评估
        # results['DBN-3层'] = accuracy

        # 可视化比较
        plt.figure(figsize=(10, 6))
        plt.bar(results.keys(), results.values())
        plt.ylabel('准确率')
        plt.title('RBM vs DBN 性能比较')
        plt.ylim(0.8, 1.0)
        plt.show()

        return results

6. 可视化层次化特征
-------------------

观察 DBN 各层学到的特征：

.. code-block:: python

    def visualize_layer_features(dbn):
        """可视化 DBN 各层的特征"""
        n_layers = dbn.num_layers

        fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 5))

        for idx in range(n_layers):
            rbm = dbn.get_rbm_layer(idx)
            weights = rbm.quadratic_coef.detach().cpu().numpy()

            # 显示权重矩阵
            ax = axes[idx] if n_layers > 1 else axes
            im = ax.imshow(weights, cmap='coolwarm', aspect='auto')
            ax.set_title(f'第 {idx+1} 层权重\n({rbm.num_visible}→{rbm.num_hidden})')
            ax.set_xlabel('隐藏单元')
            ax.set_ylabel('输入单元')
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.show()

    visualize_layer_features(dbn)

7. 完整代码
-----------

运行完整示例：

.. code-block:: bash

    cd example/dbn_digits
    jupyter notebook supervised_dbn_digits.ipynb

8. 总结
-------

.. list-table:: RBM vs DBN
   :widths: 25 35 40
   :header-rows: 1

   * - 方面
     - 单层 RBM
     - 深度 DBN
   * - 特征抽象层次
     - 单一层次
     - 多层次，逐层抽象
   * - 训练复杂度
     - 较低
     - 较高，需要逐层训练
   * - 特征表达能力
     - 有限
     - 更强大，能捕获复杂模式
   * - 适用场景
     - 简单特征提取
     - 复杂模式识别

9. 下一步
---------

- 尝试不同的层数和隐藏单元配置
- 学习 :doc:`bm_generation` 了解全连接玻尔兹曼机的生成能力
- 探索 :doc:`qvae_mnist` 了解更先进的生成模型
