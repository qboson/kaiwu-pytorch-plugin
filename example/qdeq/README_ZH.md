# 语言版本：[中文](README_ZH.md) | [English](README.md)

# `example/qdeq`

本目录提供 `QDEQ`（量子辅助深度均衡层）的示例。`QDEQ` 把一个权重共享的层
`layer(z, x)` 变成求不动点 `z* = layer(z*, x)` 的深度均衡层：前向不建计算图地求解
不动点，反向用隐函数定理计算梯度，显存占用与迭代步数无关。

前向每一步不动点更新的系数被编码成一个小规模 QUBO，交给 Kaiwu SDK 的优化器
（模拟退火或 CIM）求解；优化器由调用方创建并传入，与 `QVAE`、`EnergyModel` 的
`sampler` 用法一致。反向的隐函数梯度方程用 Anderson 加速求解。

## 背景

`QDEQ` 面向存储和位宽受限的场景，例如在边缘设备上部署的时序预测模型：

- **参数更少**：DEQ 不堆叠多个独立的层，而是反复调用同一个共享层直到收敛到不动点，
  因此只需存储一层的权重；训练显存也不随迭代步数增长。
- **更新决策离散化**：DEQ 常用的 Anderson 加速在连续实数域上计算历史迭代的组合系数。
  `QDEQ` 把每个更新系数限制在 `coeff_bits` 位的网格上，通过求解 QUBO 选出系数组合，
  更新决策本身变成低位宽的，并且可以交给组合优化后端执行，例如 Kaiwu SA 或
  Kaiwu CIM 物理求解器。
- **可与量化层配合**：不动点求解和最后一次带梯度的层调用是两个独立阶段，共享层可以
  在任一阶段使用量化形式，见[与量化模型配合使用](#与量化模型配合使用)。

## 数据

示例使用 ETTh1 电力变压器数据集（[ETDataset](https://github.com/zhouhaoyi/ETDataset)）。
下载 `ETT-small/ETTh1.csv` 后放到：

```text
example/qdeq/data/ETTh1.csv
```

使用其他位置时，通过 `--data-path` 指定。

## 快速开始

```bash
pip install -e .        # 在仓库根目录安装插件
pip install -r example/qdeq/requirements.txt
python example/qdeq/simple/simple_train_example.py --sampler sa
python example/qdeq/simple/simple_forecast_example.py --sampler sa
```

`--sampler` 可选：

- `sa`：Kaiwu 模拟退火（`kaiwu.classical.SimulatedAnnealingOptimizer`），本地运行
- `cim`：Kaiwu CIM（`CIMOptimizer` + `PrecisionReducer`），需要已授权的 Kaiwu 账号
- `anderson`：经典对照组，即 `etth1_common.py` 中的 `AndersonDEQ`，它继承 `QDEQ`
  并用 Anderson 加速重写 `solve_fixed_point`

## Simple

`simple/` 是 ETTh1 上的最小可运行示例（回看 96 步，预测 96 步）：

- `simple/etth1_common.py`：数据切分、模型、sampler 构建和 Anderson 对照组
- `simple/simple_train_example.py`：训练并保存 checkpoint
- `simple/simple_forecast_example.py`：加载 checkpoint，用指定的求解器在测试集上评估

模型参照 iTransformer：每个变量的历史序列嵌入为一个 token，编码器是一个 `QDEQ`
层，层内为一个权重共享的变量间注意力块。为了几分钟内跑完，训练窗口按步长 4 抽样，
每个 epoch 只训练 60 个 batch。

```python
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import QDEQ, QDEQConfig

sampler = SimulatedAnnealingOptimizer()
deq = QDEQ(block, sampler, QDEQConfig(f_thres=12, num_basis=4, coeff_bits=4))
z = deq(tokens)            # 与普通层一样调用
deq.get_last_stats()       # 前向/反向步数、残差、sampler 耗时
```

`simple_train_example.py` 用默认参数（2 个 epoch）在 CPU 上几分钟即可跑完，
标准化空间下的测试 MSE 约为 0.43。

## 结果：Kaiwu CIM 与 Anderson 对比

为了在物理硬件上检验 Ising 求解器，用同一个训练好的 QDEQ checkpoint，分别以
Kaiwu CIM 后端和经典的 Anderson 求解器做前向求解。设置如下：

- 模型：iTransformer（`d_model = 256`），编码器为一个共享的 `QDEQ` 层，FP32
- 数据：ETTh1，回看 96 步，预测 96 步
- 样本：按顺序取测试集前 128 个窗口（4 个 batch，每个 32）
- Ising 步：8 个搜索方向，每个系数 10 位

| 前向求解器 | MSE | MAE |
| --- | ---: | ---: |
| Anderson | 0.28730 | 0.35111 |
| Kaiwu CIM | 0.28728 | 0.35110 |

两种求解器的预测结果 MSE 相差不到 0.01%，说明通过 CIM 求得的不动点与经典求解器一致。
这里用到的 iTransformer 示例会在后续补充。

## 与量化模型配合使用

`QDEQ` 提供两个钩子，子类可以重写它们来切换共享层的状态：

- `solve_context()`：包住整个不动点求解过程，这一阶段不计算梯度
- `reattach_context()`：包住在不动点处的最后一次层调用，梯度从这里传回

例如，在重新挂接梯度时让共享层使用 W8A8 伪量化（8 位权重和 8 位激活）：

```python
class QuantizedQDEQ(QDEQ):
    def reattach_context(self):
        return self.layer.quantized()   # 开启伪量化的上下文管理器
```

这样权重只需按 8 位存储。在我们用 iTransformer 做的五个预测数据集实验中，DEQ 共享层
加上 W8A8 后，静态权重存储比显式多层模型减少 4.3×–12.8×。带 W8A8 量化的
iTransformer 示例会在后续补充。

## 求解流程

前向的每一步：

1. 计算残差 `r = layer(z, x) - z`
2. 取 `num_basis` 个搜索方向 `V`（`-r`、历史残差与历史更新）
3. 有限差分估计 `G = J V`
4. 把 `min_a ||r + G a||^2` 中的系数 `a` 用 `coeff_bits` 位二进制编码，
   经 `kaiwu.torch_plugin._kaiwu_compat.qubo_matrix_to_ising_matrix` 转成带辅助自旋的 Ising 矩阵后交给 `sampler.solve`
5. 解码最优解，更新 `z ← z + V a`
