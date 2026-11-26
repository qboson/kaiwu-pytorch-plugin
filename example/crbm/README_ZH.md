# 语言版本：[中文](README_ZH.md) | [English](README.md)

# `example/crbm`

本目录提供 `ConditionalRestrictedBoltzmannMachine`（条件受限玻尔兹曼机，CRBM）的离线可运行示例，演示基于 Kaiwu Ising 采样器的条件二值模式生成与补全流程。

## 模型简介

`ConditionalRestrictedBoltzmannMachine` 继承自 `AbstractBoltzmannMachine`，在标准 RBM 基础上引入外部条件输入 `c`（支持二值或连续特征）：

$$E(v, h \mid c) = - v^T W h - c^T U h - b_v^T v - b_h^T h - c^T A v$$

- 可通过 `use_cond_visible=True/False` 开关条件到可见层的直连权重 `A`（`cond_visible_coef`）。
- 通过将条件 `c` 折叠进有效线性偏置 `b_v_eff = b_v + c @ A` 与 `b_h_eff = b_h + c @ U`，`condition_ising_matrix(c)` 可导出与 PyTorch 能量精确等价的 Ising 矩阵，直接兼容 `SimulatedAnnealingOptimizer` 与光量子 `CIMOptimizer` 采样器。

## 快速开始

本示例完全离线运行，无需下载额外数据集或安装额外依赖。在仓库根目录执行：

```bash
pip install -e .
python example/crbm/run_crbm.py
```

可选参数：

- `--epochs`：训练轮数（默认 `35`）
- `--lr`：SGD 学习率（默认 `0.2`）
- `--seed`：随机种子（默认 `42`）
- `--sampler`：Ising 采样器后端（默认 `local` 免授权 NumPy 模拟退火；可选 `sa` 使用 Kaiwu `SimulatedAnnealingOptimizer`，或 `cim` 使用 `CIMOptimizer` + `PrecisionReducer`，需通过 `USER_ID`/`SDK_CODE` 初始化 Kaiwu 授权）

## 示例说明

`run_crbm.py` 在 4 类 8 位结构化二值模式（条件为 4 维 one-hot 类别标签）上训练 CRBM（`num_visible=8`, `num_condition=4`, `num_hidden=6`），在 CPU 上数秒即可完成并输出：

- 训练前后的**条件自由能间隔** `F(v_wrong | c) - F(v_true | c)`
- 基于 `condition_sample` 的**条件生成精确匹配率与逐位准确率**
- 面对 25% 比特翻转输入时的**条件模式补全准确率**
