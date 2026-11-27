# 文档化的训练循环真的在学习吗？——一次精确验证

Quick Start 里的训练循环是：

```python
positive = rbm.get_hidden(batch, bernoulli=True)
negative = rbm.sample(sampler)
objective = rbm.objective(positive, negative)
objective.backward()
optimizer.step()
```

`objective()` 被文档描述为"梯度等价于负对数似然梯度"的替代目标，因此取正学习率的一步必须让训练样本在模型下**更可能**出现。
本示例不去"相信"这句话，而是把它**测出来**：三可见/两隐藏的 RBM 只有 `2^5` 个状态，可以直接穷举得到某个可见模式的精确边际概率。

全程**不需要 Kaiwu SDK 许可证**——负相位使用当前模型的穷举基态（相当于一个完美的采样器），所以给定 `--seed` 后结果完全确定。

## 测什么

| 指标 | 含义 |
| --- | --- |
| `p(pattern)` | 训练模式在模型下的精确概率（三个模式取平均） |
| 能量差 | 全部 `2^5` 状态的平均能量 减去 模式能量；为正表示模式比"典型状态"更低能 |

## 运行

```bash
python example/learning_direction/learning_direction.py
```

预期输出（seed 0）：

```text
steps=200 seed=0
mean p(pattern) before: 0.124706
mean p(pattern) after : 0.132095
energy gap before     : -0.0022
energy gap after      : 0.0640
PASS: the documented loop increased the probability of the training patterns
```

概率上升约 6%，能量差由负转正——训练后模式落到平均状态之下，这正是能量模型里"学会"的含义。

## 这个检查为什么有意义

`evaluate_learning(..., flip_sign=True)` 会把目标取反。取反后同样的测量立刻崩塌：

```text
normal : p 0.124706 -> 0.132095,  gap -0.0022 -> +0.0640
flipped: p 0.124706 -> 0.000000,  gap -0.0022 -> +0.4709
```

说明 PASS/FAIL 对更新的**方向**敏感，而不是数值噪声。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `learning_direction.py` | 穷举辅助函数、确定性训练循环与命令行入口 |
| `README.md` / `README_ZH.md` | 中英文说明 |

## 依赖项

仅项目基础依赖（`torch`、`numpy`）。
