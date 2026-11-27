# Q-Diffusion 约束生成示例

本示例演示 `QDiffusion.generate(..., partial_masks=...)`：固定序列的一部分，
只让其余位置参与生成。

`partial_masks` 是与输入 token 同形状的布尔张量。标为 `True` 的位置在整个解码
过程中保持输入 token 不变；其余非特殊位置才会被提议、重排与重采样。可编辑位置
不会生成 BOS/EOS/PAD 或 mask token。

## 内容

- `constrained_generate.py`：玩具提议模型 `PreferSevenProposal`、玩具能量模型
  `PreferManySevens`，以及运行一次约束生成的 `main()`。

## 运行

```bash
python example/qdiffusion_constrained/constrained_generate.py
```

输出应为长度 6 的序列：位置 2、3 仍是输入的 `5`、`6`，其余可编辑位置是内容
token（例如 `7`）。

## 依赖

只需要 `torch` 与已安装的 `kaiwu-torch-plugin`。不需要 checkpoint、数据集下载或
Kaiwu 许可证。

## 示例校验的契约

- 固定位置逐 token 保持不变；
- 可编辑位置不会出现 BOS/EOS/PAD/mask；
- 能量模型通过 `score_conditioned` 参与候选重排。

上述不变量由 `tests/test_qdiffusion_constrained_example.py` 固定。
