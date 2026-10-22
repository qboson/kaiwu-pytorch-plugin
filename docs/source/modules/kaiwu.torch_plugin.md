# 模块手册

## Module contents

```{eval-rst}
.. automodule:: kaiwu.torch_plugin
   :members:
   :undoc-members:
   :show-inheritance:
```

## 特征选择包装器的输入布局

`FeatureSelectionWrapper` 通过 `input_feature_axis` 指定可选择特征所在的维度。
默认值 `-1` 表示最后一个维度；负索引从最后一个维度向前计数。

| 输入布局 | `input_feature_axis` |
| --- | --- |
| 单个特征向量 `(features,)` | `0` 或 `-1` |
| 二维批次 `(batch, features)` | `1` 或 `-1` |
| 转置后的二维输入 `(features, batch)` | `0` 或 `-2` |
| 序列输入 `(batch, features, steps)` | `1` 或 `-2` |

对于具有 `ndim` 个维度的输入，轴索引必须满足
`-ndim <= input_feature_axis < ndim`，并且该轴的长度必须等于 `feature_dim`。
`apply_mask` 和 `forward` 在轴越界时抛出 `ValueError`，即使其他维度恰好具有相同长度。
掩码沿其他维度广播，因此同一特征在所有样本和时间步使用相同的掩码值；
转置等非连续张量也按其逻辑形状应用掩码。
