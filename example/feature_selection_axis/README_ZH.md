# 非默认特征轴上的特征选择

`FeatureSelectionWrapper` 会遮住输入张量的某一个轴，轴由 `input_feature_axis`
指定，默认值 `-1` 对应常见的 `(batch, features)` 布局。若数据是特征在前的
`(features, batch)`，需要设置 `input_feature_axis=0`。

本示例构造两种布局，安装一个只保留特征 `0, 2, 4` 的固定掩码，并校验遮罩契约。

## 内容

- `axis_masking.py`：`FeatureFirstModel`、包装器工厂函数、掩码安装辅助函数，以及
  同时运行两种布局的 `main()`。

## 运行

```bash
python example/feature_selection_axis/axis_masking.py
```

预期输出：

```
feature-first mask: [1, 0, 1, 0, 1, 0]
zeroed feature rows: [1, 3, 5]
forward output shape: (1, 8)
mismatched feature axis rejected: input feature axis has size 5, expected 6
batch-first (axis -1) zeroed feature columns: [1, 3, 5]
axis contract OK
```

## 依赖

只需要 `torch` 与已安装的 `kaiwu-torch-plugin`。不需要数据集、checkpoint 或
Kaiwu 许可证。

## 示例校验的契约

- `input_feature_axis=0` 时，恰好未选中的特征行被置零；
- `forward` 把遮罩后的张量传给被包装的模型；
- 特征轴大小与 `feature_dim` 不一致时抛出 `ValueError`；
- 默认 `input_feature_axis=-1` 时改为置零特征列。

上述不变量由 `tests/test_feature_selection_axis_example.py` 固定。
