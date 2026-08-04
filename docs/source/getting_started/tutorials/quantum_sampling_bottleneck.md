# 为什么需要量子采样

玻尔兹曼机的训练并不只取决于模型定义；从模型分布中获得足够好的样本同样关键。本章连接理论基础与后续实践，说明采样为何会成为瓶颈，以及 Kaiwu-PyTorch-Plugin（KPP）如何让经典与相干伊辛机（CIM）采样器使用同一训练流程。

## 精确最大似然学习中的采样问题

对于基于能量的模型，负对数似然梯度可分为数据分布与模型分布上的两项期望：

$$
\frac{\partial \mathcal{L}}{\partial \theta} =
\mathbb{E}_{\mathrm{data}}\left[\frac{\partial E_\theta}{\partial \theta}\right] -
\mathbb{E}_{\mathrm{model}}\left[\frac{\partial E_\theta}{\partial \theta}\right].
$$

数据项可由 mini-batch 直接近似。模型项则要求从当前玻尔兹曼分布 $P_\theta$ 采样；精确计算通常需要枚举指数数量的构型。因此，训练质量取决于样本是否足够接近目标分布，以及取得这些样本的成本。

## 经典采样与近似训练

Gibbs 采样等 Markov Chain Monte Carlo（MCMC）方法通过局部更新逐步接近目标分布。在能量景观存在多个模态和高能垒时，链可能长时间停留在局部区域，导致混合缓慢。

对比散度（CD）以少量 MCMC 步数换取训练速度，但得到的是近似梯度；持久对比散度（PCD）通过保留链状态改善这一近似，仍需为每次参数更新付出采样成本。它们是实用方法，但不应与平衡分布的精确采样混为一谈。

## 从玻尔兹曼机到 Ising 哈密顿量

二值变量 $x_i \in \{0, 1\}$ 可通过 $s_i = 2x_i - 1$ 转换为 Ising 自旋 $s_i \in \{-1, +1\}$。相应的二次能量可写为：

$$
H(\mathbf{s}) = -\sum_i h_i s_i - \sum_{i<j} J_{ij}s_i s_j + \mathrm{const}.
$$

其中 $h_i$ 与 $J_{ij}$ 分别表示局部场和耦合。KPP 会从模型参数构造 Ising 矩阵，并将其交给 Kaiwu SDK 优化器求解；用户无需手工完成变量转换。

## KPP 中的经典与量子采样器

KPP 通过同一个 `sample(sampler)` 调用使用不同采样后端。`SimulatedAnnealingOptimizer` 适合本地调试和建立基线；`CIMOptimizer` 需要有效的 Kaiwu SDK 凭据及真机访问权限。两者的样本质量、延迟和成本应在相同模型、采样预算与评价指标下比较，不能仅由采样器类型推断优劣。

```python
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.cim import CIMOptimizer

classical_sampler = SimulatedAnnealingOptimizer()
quantum_sampler = CIMOptimizer(task_name="my_experiment", wait=True)
```

## 小结

- 玻尔兹曼机训练需要近似模型分布上的期望，采样成本是关键限制。
- MCMC、CD 与 PCD 在速度和近似误差之间取舍。
- KPP 将模型参数转换为 Ising 问题，并通过统一的采样接口连接经典与量子后端。
- 是否使用真机采样应由可用硬件、端到端延迟、样本质量和实验目标共同决定。
