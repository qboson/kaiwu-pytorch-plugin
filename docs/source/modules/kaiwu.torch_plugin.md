# 模块手册

## GBRBM 的 Gaussian 条件采样

`GaussianBernoulliRestrictedBoltzmannMachine.sample(sampler)` 默认返回外部采样器
给定 Bernoulli 状态下的 Gaussian 条件均值，保持现有确定性行为。训练负相需要
Gaussian 条件二阶矩时，使用显式随机选项：

```python
negative_states = model.sample(sampler, no_random=False)
loss = model.objective(positive_states, negative_states)
loss.backward()
```

`no_random=False` 使用已有条件采样公式 `mean + std * torch.randn_like(mean)`，
保留 `model.var` 的条件方差。两种模式均返回无梯度的完整状态，内部排列仍为
Gaussian 在前、Bernoulli 在后；随机性遵循 PyTorch 的随机种子。

完整负相的 Bernoulli 分布仍由外部采样器决定，使用随机 Gaussian 条件采样
不保证外部优化器按目标玻尔兹曼分布采样。独立的本地 Gibbs 入口保持原有行为。

## Module contents

```{eval-rst}
.. automodule:: kaiwu.torch_plugin
   :members:
   :undoc-members:
   :show-inheritance:
```
