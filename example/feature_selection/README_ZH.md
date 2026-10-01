**语言版本**：[中文](README_ZH.md) | [English](README.md)

### 神经网络上的量子特征选择

本示例演示 `FeatureSelectionWrapper` 如何在训练模型的同时筛选输入中的有效特征，筛选子问题交给 Kaiwu CIM 求解器在量子侧完成。数据集为合成数据且真值信号特征已知，可以直接对照筛选结果与 ground truth。

两个入口脚本：

* `linear_regression_solvers.py`：线性回归数据集 + **三种求解器**（`local_search`、`sa`、`kaiwu_cim`），无需量子资源即可对比不同求解器给出的特征集合；
* `neural_network_kaiwu_cim.py`：同样的流程跑在 `TinyCNN`、`SimpleRNN`、`SimpleLSTM` 三个骨干网络上，使用 `kaiwu_cim` 求解器。

运行方式：

```bash
# 运行 local_search 求解器；未配置 license 时自动跳过 sa 与 kaiwu_cim
python example/feature_selection/linear_regression_solvers.py

# sa / kaiwu_cim 路径——两者都经由 Kaiwu SDK，需要 license
export LICENSE_USER_ID="<your-user-id>"
export LICENSE_SDK_CODE="<your-sdk-code>"
export KAIWU_PROJECT_NO="<your-project-no>"   # 或直接编辑脚本中的 KAIWU_PROJECT_NO
python example/feature_selection/linear_regression_solvers.py
python example/feature_selection/neural_network_kaiwu_cim.py
```

每次运行会按模型打印：最终损失、准确率（分类）或损失（回归）、真值 `signal_features`、以及 wrapper 筛选出的 `selected_features`。在合成数据上，筛选结果应能覆盖大部分信号特征。

**文件说明**：

| 文件 | 用途 |
| --- | --- |
| `feature_selection_datasets.py` | 含已知信号特征的合成数据集（CNN 图像张量、序列、线性回归） |
| `feature_selection_models.py` | `TinyCNN`、`SimpleRNN`、`SimpleLSTM` 骨干网络 |
| `linear_regression_solvers.py` | 线性回归上的特征选择，支持 `local_search` / `sa` / `kaiwu_cim` |
| `neural_network_kaiwu_cim.py` | CNN / RNN / LSTM 上的特征选择（`kaiwu_cim`） |
| `kaiwu_license.py` | 从 `LICENSE_USER_ID` / `LICENSE_SDK_CODE` 初始化 Kaiwu license |

**依赖项**：除包自身要求外无额外依赖。`sa` 与 `kaiwu_cim` 求解器都经由 Kaiwu SDK，需要 Kaiwu license（见[安装指南](https://kaiwu-pytorch-plugin.readthedocs.io/en/latest/source/getting_started/installation.html) "Kaiwu SDK Configuration & License" 一节）；`linear_regression_solvers.py` 中的 `local_search` 与 `sa` 求解器无需 license 即可运行。
