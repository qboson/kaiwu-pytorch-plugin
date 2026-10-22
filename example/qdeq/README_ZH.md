# 语言版本：[中文](README_ZH.md) | [English](README.md)

# `example/qdeq`

本目录提供 `QDEQ`（量子辅助深度均衡层）的示例。`QDEQ` 把一个权重共享的层
`layer(z, x)` 变成求不动点 `z* = layer(z*, x)` 的深度均衡层：前向不建计算图地求解
不动点，反向用隐函数定理计算梯度，显存占用与迭代步数无关。

前向每一步不动点更新的系数被编码成一个小规模 QUBO，交给 Kaiwu SDK 的优化器
（模拟退火或 CIM）求解；优化器由调用方创建并传入，与 `QVAE`、`EnergyModel` 的
`sampler` 用法一致。反向的隐函数梯度方程用 Anderson 加速求解。

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

## 示例结果

运行 `simple_train_example.py`（默认参数，2 个 epoch）的结果，误差在标准化后的空间计算：

| 前向求解器 | test MSE | test MAE | 平均前向步数 |
| --- | ---: | ---: | ---: |
| Anderson | 0.4263 | 0.4312 | 6.3 |
| Kaiwu SA | 0.4280 | 0.4326 | 9.2 |

用 SA 训练得到的权重分别以 SA 和 Anderson 求解时，测试 MSE 均为 0.4280，说明 Ising
求解器收敛到了同一个不动点。

## 求解流程

前向的每一步：

1. 计算残差 `r = layer(z, x) - z`
2. 取 `num_basis` 个搜索方向 `V`（`-r`、历史残差与历史更新）
3. 有限差分估计 `G = J V`
4. 把 `min_a ||r + G a||^2` 中的系数 `a` 用 `coeff_bits` 位二进制编码，
   经 `kaiwu.conversion.qubo_matrix_to_ising_matrix` 转成带辅助自旋的 Ising 矩阵后交给 `sampler.solve`
5. 解码最优解，更新 `z ← z + V a`
