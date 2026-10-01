# Nemotron Contextual-Energy QDiffusion 示例

本目录把冻结的 Nemotron block-diffusion proposal、Contextual Energy Model、
官方 KPP `EnergyModel`/`BoltzmannMachine` 与 PyPI Kaiwu SA 组成一条可训练、
可推理、可断点续跑的 example-side 工作流。

代码不包含模型权重或训练数据，也不会下载这些私有产物。模型、数据、GPU 和
checkpoint 路径均由命令行显式传入。

## 固定方法边界

- 主目标：Outcome Pairwise margin loss
- 辅助项：`0.01 × NCE`，仅约束能量尺度，不称为严格 NCE
- Contextual Encoder：1024 dim、4 layers、16 heads
- KPP BM：512 visible × 256 hidden
- 推理：K=4、energy lambda=2.0、最多 8192 个新 token
- 求解器：`kaiwu.classical.SimulatedAnnealingOptimizer`

旧的自制 BM、LightSA、EDLM-NCE 和 Energy-Backbone/LoRA 路线不属于本示例。

```text
private JSONL
    │
    ▼
prepare_pairs ──► same-state OutcomePair (pairs.pt)
                         │
                         ▼
                    train.py
                         │
             Contextual Encoder + KPP BM + SA
                         │
                         ▼
                       best.pt
                         │
frozen Nemotron 原生 generate ─► ContextualEnergyHook ─► ContextualEnergyModel 打分 ─► token decision
                         │
                         ▼
                    evaluate.py
              (Native/BM matched + resume)
```

原生 Nemotron `generate` 保持 cache、去噪步骤和停止条件；`QDiffusion` 仅作为
普通的 BM 打分实例。生成时只在官方 transfer helper 产生候选、尚未写回 token 的
瞬间调用 Energy hook，因此不会复制或替代原生生成循环。

## 依赖

先安装与 CUDA 匹配的 PyTorch，再在仓库根目录运行：

```bash
pip install -e .
pip install -r example/qdiffusion/nemotron/requirements.txt
```

求解器使用仓库依赖固定的 PyPI `kaiwu` 正式版接口：

```python
from kaiwu.classical import SimulatedAnnealingOptimizer
```

Nemotron checkpoint 通过
Transformers `trust_remote_code=True` 加载，因此只能使用可信 checkpoint。

## 私有数据与 pair schema

输入是 JSONL，每行至少包含：

```json
{"problem_id":"math-train:0","split":"train","problem":"...","answer":"..."}
```

训练 pair 的 `split` 只能是 `train` 或 `val`。评测 JSONL 不要求 `split`；禁止把
benchmark test 的答案用于 pair 采集或 checkpoint 选择。同一 `problem_id` 不得跨 split。
Tensor pair 使用代码内置的 v2 schema；每条记录同时保存当前 noisy block、其 token
feature、冻结的 hidden state 和正负候选。加载和合并时检查 shape、state hash、
transfer mask、噪声到候选的变化、正负候选差异及 split 泄漏。旧 v1 pair 不能与 v2
checkpoint 混用；旧 checkpoint 也不能继续使用，需重新采集并训练。

## 入口一览

全部在仓库根目录以模块方式运行（先完成上面的安装步骤）：

| 入口 | 作用 |
|---|---|
| `prepare_pairs` | 离线采集：跑冻结模型，抓取同状态候选分支并强制走岔路，按最终答案对错标成（正，负）pair；支持分片与断点续采 |
| `merge_pairs` | 把多个独立采集的 `pairs.pt` 合并为一个工件，合并后重新校验 schema 与 train/val split 一致性 |
| `train` | 在合并后的 pair 上训练 `ContextualEnergyModel`（outcome-pairwise margin + 小权重 NCE 正则）；产出按 val ranking 选择的 `best.pt`，可通过 `last.pt` 续跑 |
| `evaluate` | 在评测 JSONL 上做 Native 与 BM 引导的对照解码，boxed-only `math-verify` 判分；按 strategy 断点续评 |

四步依次执行：`prepare_pairs` → `merge_pairs` → `train` → `evaluate`，
具体命令见下面三节。

## 1. 生成 same-state pairs

采集器按题原子保存；相同配置再次运行会跳过已完成题目。

```bash
python -m example.qdiffusion.nemotron.prepare_pairs \
  --model /path/to/Nemotron-Labs-Diffusion-8B \
  --dataset-jsonl /path/to/private_math.jsonl \
  --split train \
  --output-dir /path/to/pairs_train \
  --device cuda:0 \
  --num-candidates 4 \
  --max-new-tokens 512 \
  --block-length 8

python -m example.qdiffusion.nemotron.prepare_pairs \
  --model /path/to/Nemotron-Labs-Diffusion-8B \
  --dataset-jsonl /path/to/private_math.jsonl \
  --split val \
  --output-dir /path/to/pairs_val \
  --device cuda:0

python -m example.qdiffusion.nemotron.merge_pairs \
  --inputs /path/to/pairs_train/pairs.pt /path/to/pairs_val/pairs.pt \
  --output /path/to/pairs_merged.pt
```

只有同一 noisy state、同一 block/step、同一 transfer mask 下同时出现正确和
错误结果时才生成 pair。禁止用 benchmark test 结果采集数据或选择 checkpoint。

## 2. 训练与续跑

默认参数就是锁定配置：

```bash
python -m example.qdiffusion.nemotron.train \
  --pairs /path/to/pairs_merged.pt \
  --output-dir /path/to/contextual_energy_run \
  --device cuda:0 \
  --epochs 10 \
  --batch-size 8
```

输出：

- `run_config.json`：环境、数据计数、超参数和源码 revision
- `last.pt`：模型、优化器、scheduler、history，可继续训练
- `best.pt`：按 held-out val pair ranking 选择
- `summary.json`：最佳 epoch 和逐 epoch 指标

默认启用 `--resume`。若已有 `last.pt`，程序会比对 pair 路径/大小/SHA-256、
schema、模型及优化参数，不一致则拒绝续跑，并记录续跑环境。`--no-resume`
要求使用新输出目录。

## 3. Native/Qdiffusion 推理评测

```bash
python -m example.qdiffusion.nemotron.evaluate \
  --model /path/to/Nemotron-Labs-Diffusion-8B \
  --checkpoint /path/to/contextual_energy_run/best.pt \
  --dataset-jsonl /path/to/private_math_eval.jsonl \
  --output-dir /path/to/eval_output \
  --device cuda:0 \
  --strategies native bm \
  --num-candidates 4 \
  --energy-lambda 2.0 \
  --max-new-tokens 8192
```

每种 strategy 有独立的 JSONL partial 和配置 fingerprint。中断后相同命令继续；
checkpoint hash、数据、解码参数或 seed 改变时拒绝混用旧结果。输出记录
accuracy、tokens、NFE、耗时、完整文本和候选选择统计。

最终答案使用 boxed-only、NeMo-compatible `math-verify` 判分。
