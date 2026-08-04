# 将量子采样器集成到 PyTorch

README 中的基础示例表明，KPP 将 PyTorch 模型、优化器和 Kaiwu SDK 采样器组合在同一个训练循环中。本章说明如何将本地模拟退火替换为相干伊辛机（CIM）采样器，以及当前 KPP API 中各调用的含义。

## 核心接口

`RestrictedBoltzmannMachine` 和 `BoltzmannMachine` 均继承自 `AbstractBoltzmannMachine`。训练中最常用的接口是：

- `get_hidden(s_visible, ...)`：为 RBM 构造包含可见与隐藏状态的正相状态。
- `sample(sampler)`：根据当前模型参数构造 Ising 矩阵，调用 `sampler.solve()`，并返回负相状态张量。
- `objective(s_positive, s_negative)`：计算其梯度等价于负对数似然梯度的目标。

因此，`sample()` 不接收数据状态，也不返回可见层和隐藏层两个张量；`objective()` 只接收正相与负相两个状态张量。

## 创建 CIM 采样器

真实量子采样需要已配置的 Kaiwu SDK 授权信息和真机访问权限。`PrecisionReducer` 可用于将 Ising 参数适配到硬件精度约束。

```python
from kaiwu.cim import CIMOptimizer, PrecisionReducer

quantum_sampler = CIMOptimizer(task_name="rbm_mnist", wait=True)
sampler = PrecisionReducer(
    quantum_sampler,
    precision=8,
    truncated_precision=10,
    target_bits=550,
    only_feasible_solution=False,
)
```

## 最小训练迭代

下面的代码遵循 README 的 RBM 示例：正相状态来自数据，负相状态来自当前模型的采样结果。将 `sampler` 换为 `SimulatedAnnealingOptimizer()` 即可在本地运行同一流程。

```python
import torch
from torch.optim import SGD
from kaiwu.torch_plugin import RestrictedBoltzmannMachine

num_visible, num_hidden = 20, 30
rbm = RestrictedBoltzmannMachine(num_visible, num_hidden)
optimizer = SGD(rbm.parameters(), lr=0.01)

v_data = torch.randint(0, 2, (16, num_visible)).float()
s_positive = rbm.get_hidden(v_data, bernoulli=True)
s_negative = rbm.sample(sampler)

optimizer.zero_grad()
loss = rbm.objective(s_positive, s_negative)
loss.backward()
optimizer.step()
```

## 选择采样器

| 考虑因素 | `SimulatedAnnealingOptimizer` | `CIMOptimizer` |
| --- | --- | --- |
| 运行位置 | 本地 CPU | Kaiwu 真机服务 |
| 前提 | 无需真机凭据 | SDK 授权、访问权限与配额 |
| 适用阶段 | 调试、基线与小规模实验 | 具备真机资源后的比较实验 |
| 端到端耗时 | 由本地问题规模与设置决定 | 还包含任务提交与排队时间 |

先用经典采样器验证数据处理、模型和训练循环，再在相同实验设置下替换为 CIM 采样器，是更容易定位问题的工作方式。

## 小结

- KPP 通过 `sample(sampler)` 将采样器接入 PyTorch 训练循环。
- 负相状态由 `sample()` 返回，训练目标由 `objective(s_positive, s_negative)` 计算。
- CIM 采样应与经典基线在相同任务和评价指标下比较。
