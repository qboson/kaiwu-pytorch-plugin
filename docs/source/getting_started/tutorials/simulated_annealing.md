# 用于 Ising 模型的模拟退火

模拟退火（simulated annealing，SA）是 KPP 中可本地运行的采样基线。它既可用于寻找低能构型，也可在合适的有限温度和采样设置下，为玻尔兹曼机训练提供近似样本。

## Ising 模型与玻尔兹曼分布

Ising 模型使用自旋 $s_i \in \{-1, +1\}$ 表示变量：

$$
H(\mathbf{s}) = -\sum_i h_i s_i - \sum_{i<j} J_{ij}s_i s_j.
$$

在温度 $T$ 下，构型的概率与 $\exp(-H(\mathbf{s}) / T)$ 成正比。KPP 将玻尔兹曼机的参数转换为相应的 Ising 矩阵；采样器接收该矩阵并由 Kaiwu SDK 的 `solve()` 方法求解。

## 从优化到采样

典型的模拟退火从高温随机构型开始，反复提出翻转一个自旋。对于能量变化 $\Delta E$，Metropolis 接受概率为：

$$
P_{\mathrm{accept}} = \min\left(1, \exp\left(-\frac{\Delta E}{T}\right)\right).
$$

温度逐步降低时，算法更倾向保留低能构型。若目标是优化，可继续降温以寻找较低能量；若目标是近似玻尔兹曼采样，则必须明确温度、退火调度、独立运行次数和样本相关性。退火终点的构型不能自动视为无偏玻尔兹曼样本。

## 在 KPP 中使用 SA

KPP 的 `AbstractBoltzmannMachine.sample()` 会构造 Ising 矩阵并调用优化器。用户只需创建 `SimulatedAnnealingOptimizer` 并传入模型：

```python
import torch
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import RestrictedBoltzmannMachine

rbm = RestrictedBoltzmannMachine(num_visible=20, num_hidden=30)
sampler = SimulatedAnnealingOptimizer()
samples = rbm.sample(sampler)
```

此处 `samples` 是模型返回的状态张量；KPP 在内部调用 `sampler.solve(ising_mat)`，而不是 `sampler.sample(hamiltonian)`。

## 与真机采样的关系

SA 是本地开发、调试和基线实验的合适起点。CIM 真机采样需要凭据、任务排队和配额；是否带来收益需要针对具体模型、样本质量指标和端到端耗时进行实验比较。

## 小结

- SA 通过 Metropolis 更新和温度调度探索 Ising 能量景观。
- 优化与玻尔兹曼采样的目标不同；有限温度设置与样本评估不可省略。
- 在 KPP 中，SA 与其他优化器一样通过 `rbm.sample(sampler)` 使用。
