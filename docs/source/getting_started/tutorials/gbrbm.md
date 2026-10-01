# GBRBM：连续值数据建模

本教程演示如何使用高斯-伯努利受限玻尔兹曼机（Gaussian-Bernoulli RBM, GBRBM）对实值数据进行无监督建模与生成。当输入是连续值（传感器读数、图像灰度、物理量测量）而非二值时，应使用这一 RBM 变体。

## 目标

- 理解高斯-伯努利 RBM 与标准 RBM 的区别
- 使用对比散度在连续值数据上训练 GBRBM
- 从训练好的模型生成连续样本并评估其统计特性
- 了解无需 license 的本地 Gibbs 路径与 Kaiwu SDK 求解器路径的切换方式

## 运行环境

**示例位置**: `example/gbrbm/`

- `run_gbrbm.py`: 训练、生成与评估脚本

**依赖项**: 包自身要求（`torch`、`kaiwu`），无额外依赖。默认路径使用本地 Gibbs 采样，无需 Kaiwu SDK license。

```{code-block} bash

python example/gbrbm/run_gbrbm.py

```

## 1. 高斯-伯努利 RBM 简介

### 1.1 与标准 RBM 的区别

```{list-table}
:widths: 25 35 40
:header-rows: 1

* - 特性
  - 受限玻尔兹曼机（RBM）
  - 高斯-伯努利 RBM（GBRBM）
* - 可见单元
  - 二值 {0, 1}
  - 连续值（高斯分布）
* - 典型输入
  - 二值特征、one-hot 编码
  - 图像灰度、传感器读数、物理量
* - 可见侧能量
  - 线性偏置项
  - 对角二次型（均值 `mu` 与方差 `var`）
```

GBRBM 的一侧是连续的高斯单元，另一侧是二值的伯努利单元。`is_visible_gaussian` 控制哪一侧是高斯（默认可见侧为高斯）。模型的高斯能量在可见单元上是对角的，因此拟合目标是逐维的均值与方差。

### 1.2 能量函数

对完整状态 `s = (v, h)`（`v` 为高斯部分，`h` 为伯努利部分），能量为

$$
E(v, h) = \frac{1}{2}\sum_j \frac{(v_j - \mu_j)^2}{\sigma_j^2}
- \sum_{j,i} \frac{v_j}{\sigma_j^2} W_{ji} h_i - \sum_i b_i h_i
$$

其中 `mu`、`log_var`（方差取对数参数化）、`quadratic_coef`（即 `W`）与 `linear_bias`（即 `b`）均为可训练参数。`forward`（即 `energy`）返回该值；`marginal_energy` 将伯努利一侧求和消掉，返回高斯态的自由能。

## 2. 训练流程

训练使用对比散度（Contrastive Divergence）：

1. **正相**：`infer_from_gaussian(data)` 从数据出发，采样伯努利单元，得到完整数据状态；
2. **负相**：`gibbs_sample(n_step=1, s_gaussian=data)` 从同一批数据出发做一次模型扫描，得到模型状态；
3. **更新**：最小化 `bm.objective(s_positive, s_negative)`，其梯度与负对数似然的梯度一致，用普通 SGD 更新参数。

## 3. 生成与评估

训练完成后：

- `gibbs_sample(n_step=60, n_burnin=30, n_sample=256)` 从随机初始化的链出发，丢弃前 `n_burnin` 步热身，返回其余步骤的模型样本；
- 逐维对比数据与生成样本的均值、标准差；
- `marginal_energy` 对数据与生成样本分别计算自由能，两者接近说明模型分布与数据分布对齐。

## 4. 切换到 Kaiwu SDK 求解器

`run_gbrbm.py` 顶部的 `USE_SOLVER = True` 会将负相的本地 Gibbs 扫描替换为 `bm.sample(sampler)`：把模型伯努利一侧转成 Ising 问题并交给 Kaiwu SDK 求解。该路径需要在 [platform.qboson.com](https://platform.qboson.com) 获取凭据并调用 `kw.license.init`，与 `example/run_rbm.py` 的用法一致。

## 参考

- 模块文档: `kaiwu.torch_plugin.gbrbm`
- 示例: `example/gbrbm/run_gbrbm.py`
- 理论背景: [Theoretical Foundations](../../theoretical-foundations/index.md)
