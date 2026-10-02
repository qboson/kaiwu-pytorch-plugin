# 模块手册

## Module contents

```{eval-rst}
.. automodule:: kaiwu.torch_plugin
   :members:
   :undoc-members:
   :show-inheritance:
```

## Feature Selection 的数量约束投影

`FeatureSelectionWrapper.update_mask` 在求解后按最小/最大特征数修正候选 mask。
投影维护 QUBO 的局部收益，接近平局时重新准确求和，精确收益平局优先低索引。
完整能量中的大常数可能掩盖小收益，因此接近浮点舍入边界时，选择轨迹可能与
逐候选重算总能量不同。该贪心投影只满足数量约束，不能保证受约束的全局最优。

需要投影时，QUBO 系数应有限且形状匹配特征维度；对称化矩阵的绝对行和加对应
线性项绝对值应不超过 float64 最大值的八分之一，为计算保留数值空间。
普通无需修正的候选 mask 直接返回副本。
