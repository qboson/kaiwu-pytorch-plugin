# 最小 Q-Diffusion 示例：自带能量模型

`QDiffusion` 先生成候选序列，再用**能量模型**对候选**重排**。能量侧是公开扩展点
（`kaiwu.torch_plugin.EnergyModel`），但仓库自带示例依赖 DPLM checkpoint 与 Hugging Face
下载。本示例只保留机制本身：

| 组件 | 在本示例中的作用 |
| --- | --- |
| `ToyProposal` | 一张 embedding 表，把每个 token 映射成固定的 logits 行 |
| `BagOfTokensEnergy` | `EnergyModel` 子类：候选的 token 权重之和 减去 丢弃已观测上下文的惩罚 |
| `SequenceTokenSpec` | 六词表的 padding/BOS/EOS/mask 编号 |
| `QDiffusion.generate` | 对部分被掩码的序列执行迭代解码 |

脚本会报告能量模型被调用了几次、以及掩码是否全部解码，让"重排"这一步可见而不是隐式发生。

## 运行

```bash
python example/qdiffusion_minimal/qdiffusion_minimal.py
```

预期输出（seed 0）：

```text
steps=4 seed=0 context_weight=1.5
generated tokens        : [1, 4, 4, 4, 4, 2]
energy-model calls      : 4
ordinary-token fraction : 1.00
PASS: the energy model reranked every candidate and the mask was fully decoded
```

`--steps` 控制解码步数，`--seed` 保证可复现，`--context-weight` 调整能量模型对"保留已观测
token"的奖励强度。

## 编写自己的能量模型

继承 `EnergyModel` 并实现 `score_conditioned(noisy_tokens, candidate_tokens,
attention_mask)`，为每个候选返回形状 `[batch, 1]` 的能量；重排步骤偏好**更低**的能量。
`qdiffusion_minimal.py` 里的 `BagOfTokensEnergy` 就是一个完整例子。

## 依赖项

仅项目基础依赖（`torch`）；不需要 checkpoint、不需要 Hugging Face 下载、不需要 Kaiwu SDK
许可证，也不需要 GPU。
