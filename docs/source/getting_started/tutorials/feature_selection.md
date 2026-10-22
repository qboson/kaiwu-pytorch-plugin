# 量子特征选择：FeatureSelectionWrapper

本教程演示如何使用 `FeatureSelectionWrapper` 在训练神经网络的同时筛选输入特征：特征选择子问题被表述为 QUBO，交给 `local_search`、`sa` 或 Kaiwu CIM 求解器求解。数据集为合成数据且真值信号特征已知，可以直接评估筛选质量。

## 目标

- 理解 `FeatureSelectionWrapper` 的训练-筛选交替流程
- 对比本地求解器（`local_search` / `sa`）与量子求解器（`kaiwu_cim`）
- 在已知 ground truth 的数据上评估筛选结果

## 运行环境

**示例位置**: `example/feature_selection/`

- `linear_regression_solvers.py`: 线性回归 + 三种求解器（无 license 即可运行 `local_search` / `sa` 部分）
- `neural_network_kaiwu_cim.py`: CNN / RNN / LSTM + `kaiwu_cim`

**依赖项**: 包自身要求，无额外依赖。`kaiwu_cim` 需要 Kaiwu license（`LICENSE_USER_ID` / `LICENSE_SDK_CODE` 环境变量）。

```{code-block} bash

python example/feature_selection/linear_regression_solvers.py

```

## 1. 工作流程

`FeatureSelectionWrapper` 包装任意 PyTorch 模型，在输入侧附加 0/1 特征掩码：

1. **权重训练**：按 `mask_update_epochs` 设定的间隔，用被掩码后的输入正常训练模型权重；
2. **掩码更新**：把"哪些特征对损失重要"表述为 QUBO 问题（线性项来自各特征对损失的贡献，二次项编码特征间的冗余/配对关系），交给所选求解器；
3. **筛选输出**：`selected_indices()` 返回求解器选中的特征集合。

## 2. 评估方式

示例数据集在构造时就指定了 `signal_features`（真正携带标签信息的特征下标）。每个运行结束时打印：

```text
{'model': 'cnn', 'loss': ..., 'accuracy': ...,
 'signal_features': [...], 'selected_features': [...]}
```

`selected_features` 与 `signal_features` 的重合度即为筛选质量的直接度量。

## 3. 求解器选择

| 求解器 | 需要 license | 适用场景 |
| --- | --- | --- |
| `local_search` | 否 | 快速验证、小规模特征 |
| `sa` | 否 | 经典模拟退火基线 |
| `kaiwu_cim` | 是 | 相干伊辛机，大规模/高质量筛选 |

`solver_kwargs` 透传给对应求解器（如 CIM 的 `target_precision` / `max_bits` / `project_no`，见 `maifs.qubo.solve_qubo` 文档）。

## 参考

- 模块文档: `kaiwu.torch_plugin.maifs`
- 示例: `example/feature_selection/`（含双语 README）
- 安装与 license: [Installation Guide](../installation.md)
