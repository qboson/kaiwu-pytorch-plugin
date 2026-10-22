# 向 QVAE 注入自定义组件

`QVAE` 通过 `_create_encoder`、`_create_decoder`、`_create_bm`、`_create_sampler`
构建缺失的组件；构造器里显式传入的组件优先，因此子类可以复用已有的编码器、
解码器、玻尔兹曼机或采样器，而不必重新创建。

本示例把四个玩具组件接入一个小子类，并端到端校验契约。

## 内容

- `custom_qvae.py`：`ToyEncoder`、`ToyDecoder`、`ConstantSampler`、
  `InjectedQVAE`，以及执行一次前向、一次 loss 反向、一次能量计算的 `main()`。

## 运行

```bash
python example/qvae_custom_components/custom_qvae.py
```

预期输出：

```
recon: (5, 6) q: (5, 4) zeta: (5, 4)
encoder sees mean-centered input: True
loss: 4.0923 encoder grads: [True, True]
energy shape: (5,)
custom components OK
```

## 依赖

只需要 `torch`、`numpy` 与已安装的 `kaiwu-torch-plugin`。不需要数据集、
checkpoint 或 Kaiwu 许可证。

## 示例校验的契约

- 构造器保留传入的 encoder、decoder、BM 与 sampler；
- `forward` 返回 `(重构 logits, 后验, 潜变量 logits, zeta)` 且形状符合预期；
- 设置 `set_dataset_mean` 后，编码器收到的是 `x - mean`；
- `loss` 有限，且梯度同时回传到 encoder 与 decoder；
- `energy` 在 Bernoulli 硬化后的潜状态上计算，每个样本返回一个值。

BM 必须覆盖潜维：`num_visible + num_hidden` 等于 `config.num_latent_units`，
因为 `loss` 与 `energy` 都是在完整潜向量上评估 BM。

上述不变量由 `tests/test_qvae_custom_components_example.py` 固定。
