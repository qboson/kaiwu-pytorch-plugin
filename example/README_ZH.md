**语言版本**：[中文](README_ZH.md) | [English](README.md)

### BoltzmannMachine 与 RestrictedBoltzmannMachine 的使用示例

* `run_bm.py`：演示如何使用 `BoltzmannMachine` 类进行模型实例化、采样、目标函数计算和参数优化。适用于理解玻尔兹曼机（Boltzmann Machine）的基本训练流程，包括采样和梯度反向传播。
* `run_rbm.py`：演示如何使用 `RestrictedBoltzmannMachine` 类进行训练，包括隐藏特征提取、采样、目标函数计算和参数优化。适用于理解受限玻尔兹曼机（Restricted Boltzmann Machine, RBM）的典型应用流程。

这两个脚本均完整展示了模型初始化、采样、目标函数计算、梯度下降和参数更新的全过程，可作为快速上手玻尔兹曼机相关模型的参考示例。

---

### 分类任务：手写数字识别

本节提供了两种互补的方法，分别基于不同的神经网络架构实现手写数字识别，均展示了无监督特征学习在分类任务中的作用。

#### 基于 RBM 的分类方法

该示例演示了如何在手写数字数据集（Digits）上使用受限玻尔兹曼机（RBM）进行特征学习与分类。适合初学者理解 RBM 在图像特征提取与分类中的应用流程，并可作为后续进阶实验与扩展的基础。主要内容包括：

* **数据增强与预处理**：将原始 8×8 手写数字图像分别向上、下、左、右平移以扩充数据集，并使用 MinMaxScaler 对特征进行归一化；
* **RBM 模型训练**：实现 `RBMRunner` 类以封装 RBM 的训练过程，支持在训练过程中可视化生成样本和权重矩阵；
* **特征提取与分类**：训练完成后，利用 RBM 隐藏层的表示作为特征，输入逻辑回归模型进行分类；
* **可视化与分析**：支持在训练过程中生成样本并可视化权重，有助于观察和评估模型的学习效果。

请通过 `example/rbm_digits/rbm_digits.ipynb` 运行该示例。

#### 基于 DBN 的分类方法

该示例在 RBM 方法的基础上，进一步构建了一个完整的深度信念网络（Deep Belief Network, DBN），包含多层 RBM，提供更复杂的特征学习能力和灵活的训练策略。该实现可视作 RBM 方法的直接演进，展示了如何通过堆叠多个 RBM 来学习输入数据中越来越抽象的表示。主要内容包括：

* **层次化特征抽象**：每一层 RBM 学习对下一层激活模式的表示，形成“特征的特征”，从而捕捉数据中日益抽象的规律；
* **无监督预训练**：通过 `DBNPretrainer` 类实现逐层贪婪的 RBM 训练，使用对比散度（Contrastive Divergence）算法；
* **双重训练策略**：
  - *微调模式（Fine-tuning Mode）*：预训练后，使用 `SupervisedDBNClassification` 对整个网络进行端到端的反向传播微调；
  - *分类器模式（Classifier Mode）*：在 DBN 提取的特征上使用传统机器学习分类器；
* **高级架构设计**：基于 PyTorch 实现，并通过 `AbstractSupervisedDBN` 基类提供与 scikit-learn 的兼容性。

请通过 `example/dbn_digits/supervised_dbn_digits.ipynb` 运行该示例。

**依赖项**

```
scikit-learn
matplotlib
scipy
```

### 生成任务: 基于BM的数据生成
展示利用对比散度等方法对全连接的玻尔兹曼机进行无监督训练，从而生成与训练数据统计特性相似的新样本。该案例适合用于快速生成大量的小规模样本，且训练数据和模型规模较小的场景。

**模型构建**: 使用KL散度和NCL共同训练BM  
**训练流程**: 实现Trainer，实现学习率的scheduler和不同采样阶段的分别采样  
**数据可视化**：对采样结果的分布进行可视化  

通过`example/bm_generation/train_bm.ipynb`运行训练代码，通过`example/bm_generation/sample_bm.ipynb`运行测试代码
**依赖**
```
kaiwu==1.3.0
pandas
matplotlib
```

---

### 生成任务：基于Q-VAE的MNIST图像生成

该示例演示了如何在 MNIST 手写数字数据集上训练和评估量子变分自编码器（Quantum Variational Autoencoder, Q-VAE）模型。适用于希望了解 Q-VAE 模型训练、生成与评估流程的用户，并可作为生成模型进一步研究的基础。主要内容包括：

* **数据加载与预处理**：标准化加载 MNIST 数据集，结合展平（flatten）变换，并支持 GPU 加速；
* **模型构建**：搭建 Q-VAE 架构，包括编码器与解码器模块，以及基于 RBM 的潜在变量建模；
* **训练流程**：设计并实现完整的训练循环，跟踪损失函数、证据下界（ELBO）、KL 散度等指标，并支持检查点保存；
* **可视化与生成**：提供原始图像、重建图像与生成图像的并排可视化，直观评估模型性能。

请通过 `example/qvae_mnist/train_qvae.ipynb` 运行该示例。

**依赖项**

```
torchvision==0.22.0
torchmetrics[image]
```

---

### 表征学习：基于QVAE的潜在特征提取与分类

该扩展示例展示了如何利用预训练的 Q-VAE 表示来完成下游分类任务，体现了 Bengio（2013）提出的“学习数据的表征，使得在构建分类器或其他预测器时更容易提取有用信息”的核心思想。主要内容包括：

* **无监督特征学习**：Q-VAE 编码器在无标签监督的情况下学习有意义的特征；
* **迁移学习**：预训练表征能够高效适配下游任务；
* **多任务能力**：同一套表征同时支持生成任务与分类任务；
* **模型可解释性**：通过 t-SNE 可视化对潜在空间结构进行定性评估，提供关于类别分离性和训练过程中聚类形成的深入洞察。

请通过 `example/qvae_mnist/train_qvae_classifier.ipynb` 运行该示例。

**依赖项**

```
torchvision==0.22.0
```