# 自带采样器（Bring Your Own Sampler）

本示例展示 `Kaiwu-PyTorch-Plugin` 依赖的**采样器契约**，并用穷举真值验证一个自写的
求解器，因此无需 Kaiwu SDK 许可证即可在本机跑通这套接口。

## 插件对 `solve()` 的要求

`model.get_ising_matrix()` 返回 `(N + 1) x (N + 1)` 的辅助自旋矩阵 `M`。模型能量满足
`E(x) = c - y^T M y`（其中 `y = [2x - 1, +1]`），所以**低能量（高概率）状态就是
`s^T M s` 的最大值点**——这正是 Kaiwu SDK 求解器采用的约定。因此采样器必须：

1. 接受该 NumPy 矩阵，返回形状为 `(num_reads, N + 1)` 的数组；
2. 自旋只能取 `-1` 或 `+1`；
3. 返回的状态应最大化 `s^T M s`（等价于最小化模型能量）。

插件用 `spins[:-1] * spins[-1]` 解码，并映射回模型的 `0/1` 状态。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `local_sampler.py` | `MetropolisSampler`（纯 PyTorch 模拟退火）+ 验证与训练冒烟测试 |
| `README.md` / `README_ZH.md` | 中英文说明 |

## 运行

```bash
python example/local_sampler/local_sampler.py
```

脚本构造一个具有三个简并基态的四节点玻尔兹曼机，穷举全部 `2^4` 个状态，再通过
`model.sample(sampler)` 采样，把解码结果与穷举最小值对照：

```text
ground states (exhaustive): [(0, 0, 1, 1), (0, 1, 1, 1), (1, 0, 1, 1)]
ground energy             : -1.5000
decoded samples           : [(0, 1, 1, 1), (1, 0, 1, 1), ...]
minimal-energy draws      : 32/32 -> PASS
training smoke test       : objective 0.0377 -> 0.1591 over 5 step(s)
```

`--steps`、`--reads` 分别控制训练冒烟测试步数与每次 `solve()` 的采样条数，`--seed`
用于复现实验。

## 在自己的训练循环中使用

```python
from local_sampler import MetropolisSampler

sampler = MetropolisSampler(num_reads=8, num_sweeps=100, seed=0)
negative = rbm.sample(sampler)          # 解码后的 0/1 状态
```

采样器接口是鸭子类型的，同一个对象可直接用于 `RestrictedBoltzmannMachine`、
`BoltzmannMachine` 与 `GBRBM`——只需要实现 `solve()`。

## 依赖项

仅需项目基础依赖（`torch`、`numpy`），**不需要** Kaiwu 许可证。
