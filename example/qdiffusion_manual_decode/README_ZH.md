# 手动驱动 Q-Diffusion 解码

`QDiffusion.generate` 在内部跑完整个解码循环。同一循环也可以由外部驱动：用
`initialize_state` 创建解码状态，每次 `step` 跑一轮“提议 / 重排 / 重新掩码”，
往 `state["history"]` 追加一帧，并返回新状态，不修改传入的旧状态。

本示例手动驱动循环、在不再有 mask token 时提前停止，并校验同样种子下结果与
`generate` 一致。

## 内容

- `manual_decode.py`：`PreferSevenProposal`、`PreferManySevens`、
  `build_generator()`、`masked_positions()`，以及包含循环的 `main()`。

## 运行

```bash
python example/qdiffusion_manual_decode/manual_decode.py
```

预期输出：

```
initial masked positions: 2
step 1: masked=1 history=2
step 2: masked=1 history=3
step 3: masked=1 history=4
step 4: masked=0 history=5
final tokens: [[0, 7, 5, 6, 7, 1]]
manual loop matches generate: True
manual decode OK
```

## 依赖

只需要 `torch` 与已安装的 `kaiwu-torch-plugin`。不需要 checkpoint、数据集或
Kaiwu 许可证。

## 示例校验的契约

- `initialize_state` 从 step 0 开始，history 只有一帧；
- 每次 `step` 让 `step` 加一、追加一帧 history，并返回新状态（旧状态不变）；
- `partial_masks` 固定的位置保持输入 token；
- 足够多步之后所有 mask token 都被解码；
- 同一种子下，手动循环与 `generate` 输出一致。

上述不变量由 `tests/test_qdiffusion_manual_decode_example.py` 固定。
