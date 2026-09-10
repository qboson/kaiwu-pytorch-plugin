# nemotron 文件与流程导览

本文档逐个文件讲解 nemotron 示例:每个文件是什么、里面有哪些类和函数、各自干什么;
最后说明整条流水线如何串起来。命令与参数细节见 [README_ZH.md](README_ZH.md),
逐文件改动记录见 [CHANGES.md](CHANGES.md)。

---

## 0. 这个示例是什么

用**可学习的 Boltzmann Machine 能量模型**(KPP BM 512×256,Kaiwu SA 求解)引导
**冻结的 Nemotron block-diffusion LLM** 做数学推理解码:LLM 按自己的原生循环生成,
能量模型只在每个 block 的"改写决策点"上被咨询——它认为有显著更好的候选就覆盖,
否则放行。LLM 权重全程冻结,只训练能量侧。

对 main 的全部改动都是为支撑这个示例,逐文件一行:

| 文件(main → 本分支) | 改动 |
|---|---|
| `src/.../qdiffusion.py` | `score_visible_logits(num_lowest=...)`;`train()` 冻结保护;删除死 API(weight/temperature/history);`SequenceTokenSpec` 导出;全量 Google docstring |
| `src/.../__init__.py` / `_qdiffusion_sampling.py` | 前者导出 `SequenceTokenSpec`;后者仅文件头 |
| `tests/` 三个文件 | `score_visible_logits(num_lowest)`、`train()` 冻结保护、导出断言等契约测试 |
| `dplm/models/esm_patch.py` | 重写为 SDPA 实现(镜像 4.39.2 签名) |
| `dplm/utils/runtime.py` | checkpoint 版本字段 + 原子写 + 加载校验 |
| `dplm/` 其余 + `simple/` | 入口收敛(删 2 个 re-export 壳 + 2 个 bootstrap)、import 拍平、docstring 补齐 |
| `example/qdiffusion` 根 | README ×2 更新、requirements 增 fair-esm |
| 仓库根 | `.gitignore`、pylintrc |
| 删除 | `train_workflow.py`、`eval_esm2_distances.py`、`_example_bootstrap.py` ×2 |

细节见 [CHANGES.md](CHANGES.md) 第二节。

---

## 1. `common/` —— 共享地基

### `common/runtime.py`(108 行)

| 符号 | 作用 |
|---|---|
| `read_jsonl(path)` | 读评测/采集共用的 JSONL;逐行校验必须是含 `problem`/`answer` 键的对象(带行号报错),空数据集报错 |
| `prompt_ids(tokenizer, problem, device)` | 共享指令模板("把最终答案放进 `\boxed{}`")+ 题面 → 走 chat template → `[1, seq_len]` 张量 |
| `load_nemotron(path, device)` | `trust_remote_code=True` 加载冻结 Nemotron,`bfloat16` + eval 模式 |
| `file_identity(path)` | `path/size/sha256` 三元组——断点续跑指纹的原料 |
| `atomic_json(path, payload)` / `atomic_torch_save(path, payload)` | tmp 文件 + `os.replace` 的原子写,崩溃不产生半截产物 |
| `native_generate(session, inputs, ...)` | 原生生成的**唯一入口包装**:`temperature=0.0` 在此写死,保证所有 CLI 的原生解码行为逐比特一致 |

### `common/answers.py`(98 行)

| 符号 | 作用 |
|---|---|
| `last_boxed_content(text)` | 取最后一个完整的 `\boxed{...}` 内容,支持嵌套 LaTeX 大括号(深度计数);无完整组返回 `None` |
| `require_math_verify()` | 返回已安装的 `math-verify` 版本;未安装则运行前直接失败(判分是权威环节,不许静默降级) |
| `math_equivalent(prediction, gold)` | 三级判分:① 选项字母(A–J)走 `StringExtractionConfig`;② LaTeX 归一化后精确比对;③ `math_verify` 的 `parse/verify` 兜底 |
| `_additional_math_verify_normalize` | 判分前的补充归一化(去百分号、尾部句点) |

### `common/pairs.py`(214 行)

训练数据的**schema 与校验层**。

| 符号 | 作用 |
|---|---|
| `PAIR_SCHEMA_VERSION = 2` / `VALID_SPLITS` | schema 版本与合法 split 集合;v1 产物与 v2 checkpoint 互不兼容 |
| `OutcomePair`(frozen dataclass,16 字段) | 一条同状态反事实对:同一 `state_hash` + 同一 `transfer_mask` 下,`positive_tokens`(答对)与 `negative_tokens`(答错),外加 `noisy_tokens`、`hidden_states`、三组 token 特征、正负 proposal logprob、`negative_kind` |
| `validate_pair(record, index)` | 逐条硬校验:17 个必需字段、schema 版本、split 合法、state_hash 必须是小写 SHA-256、8 个张量字段类型/维度/形状一致、transfer_mask 为 bool 且至少一个 True、transfer 位置上候选 ≠ noisy、正负候选必须不同 |
| `validate_problem_splits(records)` | 同一 `problem_id` 不得跨 split(防评测泄漏) |
| `load_pairs(path)` / `save_pairs(path, records)` | `torch.load(weights_only=True)` + 全量校验后返回;保存走原子写。合并分片时同一对函数重跑,天然覆盖跨分片泄漏检查 |

---

## 2. `models/` —— 能量模型

### `models/energy.py`(394 行)

| 符号 | 作用 |
|---|---|
| `ContextualEnergyModel(EnergyModel)` | 本示例的能量模型,结构:三个投影器(hidden/candidate/noisy)→ 位置嵌入 → `TransformerEncoder`(norm_first + GELU)→ 池化 → `contextual_to_visible` 得到 BM 可见单元 logits;BM 隐藏单元由 Kaiwu SA 采样(`sample_hidden_state`),能量取最低 `energy_num_lowest` 个解的均值 |
| `__init__(...)` | 9 个超参的正数守卫(合并报错,直接列出违规参数名;含 BM 维度与采样数)+ 整除校验(以 ValueError 前置,因 PyTorch 只抛 AssertionError 且 `-O` 会剥离);超参存入实例供 `get_config` 序列化;`super().__init__` 接 BM 维度与 SA 采样器;末尾 `self.to(device)` + `energy_bm.device` 手工同步(指向 `fix/bm-device-sync` 的 TODO workaround) |
| `discretize_visible_state(logits)` | **override 为 identity**:可见条件保持连续值(与 KPP NLP 工作流一致),梯度无二值化损失 |
| `build_visible_logits(hidden_states, noisy_features, candidate_features, mask)` | 上下文编码主路:三路投影 → 位置嵌入 → Transformer → 池化打分 → 投影到 `[batch, bm_num_visible]` |
| `score_conditioned(noisy, candidate, mask, *, hidden_states, noisy_features, candidate_features)` | 打分入口:三个上下文必须齐(否则 ValueError);`changed = noisy != candidate`,transfer 之外的特征**替换为 noisy 特征**(逼模型只看编辑位置);调 `build_visible_logits` → `score_visible_logits(num_lowest)` |
| `get_config()` / `compact_state_dict()` / `load_compact_state_dict(state)` | versioned 配置(含 `checkpoint_format`、`visible_transform=identity`、`feature_mode` 标识)与 9 个模块的紧凑 state 读写;加载时缺失/多余键都报错 |
| `model_from_config(config, sampler, device)` | 从 checkpoint 的 `model_config` 字典重建模型 |

### `models/checkpoint.py`(119 行)

| 符号 | 作用 |
|---|---|
| `checkpoint_payload(model, epoch, energy_mean, energy_std, energy_lambda, metrics, run_config, training_state=None)` | 组装版本化 payload:`checkpoint_format`、`model_config`、`model_state`(紧凑)、`energy_normalization`(mean/std)、`energy_lambda`、指标、可选训练状态 |
| `save_checkpoint(path, payload)` / `load_checkpoint(path, device, sampler)` | 原子保存;加载时校验 payload 类型、`checkpoint_format`、必需字段,重建模型并 `load_compact_state_dict`,返回 `(model, payload)` |

---

## 3. `generation/` —— 与冻结 LLM 的桥接

### `generation/proposal.py`(308 行)

| 符号 | 作用 |
|---|---|
| `ProposalDecision`(frozen) | 一次覆盖决策:`tokens` + `transfer_index` |
| `ProposalStep`(frozen) | 一个决策点的完整快照:`block_index/step_index/nfe`、`block_tokens`、`sequence_tokens`、`hidden_states`、`native_decision`——hook 和候选生成的全部输入 |
| `NativeGenerationSession` | 上下文管理器,包装冻结模型**原生的 generate**:cache、块级去噪节奏、停止条件全部由原生循环管理。hook 在**构造期注入**(`__init__(proposal_hook=...)`),`generate(...)` 驱动循环并在每个 transfer 决策点(候选已产生、token 未写回)回调 `self.proposal_hook(step)`;`_capture_hidden_state` 抓冻结隐状态供上下文打分;`_validate_decision` 校验 hook 返回决策的合法性 |
| `close()` / `__enter__ / __exit__` | 释放原生资源 |

### `generation/candidates.py`(208 行)

| 符号 | 作用 |
|---|---|
| `_sample_k_candidates(logits, num_candidates, temperature, noise_scale)` | 对一个 block 的 proposal logits 做 Gumbel 采样,产出 `[K, block]` 的候选与逐位置分数 |
| `_mask_logits(logits, mask_id)` | 克隆并压制 mask token 的 logits |
| `_gather_transfer_positions(candidate, block_tokens, transfer_index)` | transfer 位置取候选 token、其余位置取原生 token——候选只在"该改的地方"与原生不同 |
| `LogProbScorer.score(step, candidates)` | 候选的平均 proposal logprob(调用方再按 transfer 数归一) |
| `GumbelNoiseGenerator.generate_hybrid(step, num_candidates)` | **第 0 行永远是原生选择**,其余行为采样结果——保证候选集里"保留原生"与"尝试替代"并存 |
| `build_diverse_transfer_candidates(step, raw, num_candidates)` | 按 transfer 位置内容去重,保证 K 个候选彼此不同;返回候选与多样性统计 |

### `generation/guidance.py`(316 行)

| 符号 | 作用 |
|---|---|
| `ContextualEnergyHook.__init__(qdiffusion, ...)` | 组装顾问:K、能量归一化参数、`energy_lambda`、两级保守阈值(`min_residual_gain`/`min_energy_gain`)、`guidance_end_fraction`(block 后段不再干预)、`candidate_mask_policy`(只打分 transfer 位还是整块);`token_feature_provider` 取自 proposal 的 input embeddings |
| `ContextualEnergyHook.__call__(step)` | 决策点四步:① `generate_hybrid` 造候选(原生排第 0 行)+ 多样性筛选;② 特征展开后**直达** `qdiffusion.energy_model.score_conditioned` 具名打分(上下文是能量模型的能力,不走通用门面);③ `residual = proposal_logprob − λ·归一化能量`,取最优后过四道保守门槛;④ 追加 stats 留痕。返回 `None`(放行)或 `ProposalDecision`(覆盖) |
| `get_stats()` | 导出决策统计(evaluate 报告的"候选选择统计"来源) |
| `build_nemotron_qdiffusion(proposal_model, energy_model, ...)` | 组装 QDiffusion 宿主:冻结 proposal + 能量模型 + token_spec + config(构造即冻结) |
| `load_nemotron_guidance(proposal_model, checkpoint_path, ...)` | 从 checkpoint 重建能量模型(载入归一化统计与 λ)并接好 hook,返回 `(qdiffusion, selection_hook)` |

---

## 4. 入口脚本(仓库根目录 `python -m` 运行)

### `prepare_pairs.py`(约 656 行)—— 训练数据采集

| 符号 | 作用 |
|---|---|
| `CapturedCandidateStep`(frozen)+ `best_alternative_penalty` | 一次捕获机会的 CPU 快照(候选/分数/mask/state_hash/可选隐状态);penalty = 原生分与最强替代分之差,越小代表决策越"摇摆" |
| `CandidateTraceHook` | 采集阶段挂在原生循环上:每个决策点记录完整快照(Gumbel 候选 + 打分 + 多样性筛选);`ranked_branch_points(limit)` 每 block 留 penalty 最小的点,全局升序截断 |
| `ForcedCandidateHook(captured_step, candidate_index)` | 强制岔路:重放原生生成,逐点校验 `transfer_index` 与目标一致(提前分叉/重复到达都 RuntimeError),到达后强制返回指定候选 |
| `_problem_id` / `_rendered_config` / `_load_state` / `_save_state` | 题目标识(显式 id 或题面哈希)、run_config 指纹、`collector_state.pt` 的断点续采与校验 |
| `_decode(tokenizer, output, prompt_len, gold)` | 解码续写 → `last_boxed_content` 提取 + `math_equivalent` 判分 → `{prediction, reward, final_text}` |
| `_materialize_pairs(...)` | 正样本 × 负样本(按 logprob 降序)两两组合成 pair 记录 |
| `main` | 逐题:原生跑一遍(trace hook 同步抓取)→ 判对错 → 按对错取不同数量的 branch points → 逐个 forced rollout → 有对有错才配对 → 校验后落 `pairs.pt`;逐题原子保存 state,`--resume` 续采 |

### `merge_pairs.py`(24 行)

`--inputs` 多个 `pairs.pt` → 逐个 `load_pairs`(加载即校验)→ 拼接 → `save_pairs`(含跨分片 split 泄漏复查)。

### `train.py`(623 行)—— 能量模型训练

| 符号 | 作用 |
|---|---|
| `PairDataset` / `collate_pairs` / `to_device` | 内存数据集;张量堆叠 + 标量/列表字段;按字段搬设备 |
| `score_pair_batch(model, batch)` | 正、负候选各调一次 `score_conditioned`(具名上下文) |
| `pair_loss(pos, neg, margin, nce_weight)` | `relu(margin + E⁺ − E⁻)` 的 outcome-pairwise 主目标 + `0.01×(softplus(E⁺)+softplus(−E⁻))` 的**尺度正则**(只锚能量量级,防归一化失稳;不称为严格 NCE) |
| `evaluate_rows(model, loader, device)` | eval 模式下过全部 pair,产出逐对能量/logprob 行 |
| `energy_normalization(rows)` | 能量 mean/std(**只用 train 行,无 val 泄漏**);推理时以此为归一化 |
| `summarize_rows(rows, normalization, energy_lambda)` | `ranking_accuracy`、`residual_ranking_accuracy`(**重放推理决策规则**)、能量差分位数、按 `negative_kind` 分组 |
| `_compatibility_config(args, first_record)` | 维度/超参指纹:续跑时与 `last.pt` 严格比对,不匹配即拒绝 |
| `_build_loaders(records, args, generator)` | 按 split 建 DataLoader;train 的 shuffle 用独立 `Generator(seed+1)` |
| `_train_epoch(model, loader, optimizer, ...)` | 前向 → pair_loss → 反向 → 梯度裁剪 → step;统计 loss/两分量/胜率 |
| `main` | 种子全设 → 加载 pairs → resume(校验指纹,恢复优化器/scheduler/history/best/**全部 RNG 状态**)→ AdamW + `ReduceLROnPlateau(max, on val ranking)` → 每 epoch:训练 → train/val 评估 → 归一化(仅 train)→ 汇总 → **原子写 `last.pt`(每轮)**,提升才写 `best.pt` → 早停 → `summary.json` |

### `evaluate.py`(285 行)—— Native/BM 对照评测

| 符号 | 作用 |
|---|---|
| `parse_args` | `--strategies native bm`、`--num-candidates`(原 `--K`)、`--energy-lambda`、`--resume` 等 |
| `_file_identity` / `_fingerprint` | 数据/checkpoint 的 name+size+sha256 身份;配置 → 指纹哈希 |
| `_partial_store(args, strategy)` | 每个 strategy 一套 `{strategy}.meta.json` + `{strategy}.items.jsonl`;meta 记指纹,存在则校验 `--resume` 与指纹一致 |
| `_load_partial_items(path)` | 读 append-only JSONL;**自动修复崩溃留下的撕裂尾行**(truncate + fsync) |
| `_append_item(path, item)` | 逐题追加,写后 flush + fsync |
| `_summarize(strategy, items, fingerprint)` | accuracy/平均 token/平均 NFE/耗时 + 完整 items |
| `main` | 逐题双 strategy 对照:native 走纯原生,bm 走 `load_nemotron_guidance` 的 hook;逐题 `_append_item` 落盘,中断后同命令续跑 |

---

## 5. 怎么串起来

### 5.1 训练数据与模型的生产线(离线,四阶段)

```
① 采集  python -m ...prepare_pairs   每题:原生解题 + 抓决策点 + 强制岔路 + 判题
        产物 pairs.pt:同 state_hash、同 transfer_mask 下,对/错候选各一
                │
② 合并  python -m ...merge_pairs     多分片 → 一份,复查 schema 与 split 泄漏
                │
③ 训练  python -m ...train           margin ranking 学"同状态下正比负能量低"
        产物 best.pt(能量模型 + 归一化统计 + λ)
                │
④ 评测  python -m ...evaluate        native vs bm 同题对照,math-verify 判分
```

### 5.2 推理时序(在线,一次 decode 内)

```
frozen Nemotron generate(原生循环:cache/去噪/停止全部原生管理)
   │ 每个 transfer 决策点,写回前:
   ▼
CandidateTraceHook / ContextualEnergyHook.__call__(step)
   ① GumbelNoiseGenerator.generate_hybrid → 原生 + 采样候选
   ② build_diverse_transfer_candidates → 去重保多样性
   ③ score_conditioned(tokens, candidate, mask,
        hidden_states=冻结隐状态, features=...)   ← 能量打分
   ④ residual = logprob − λ·(E − mean)/std
        ├─ 没有候选显著更好 → None,原生照写
        └─ 有 → ProposalDecision 覆盖写回
```

evaluate 的两种 strategy 就是这个注入开关:`native` 构造无 hook 的会话
(`NativeGenerationSession(model)`),`bm` 注入 selection_hook——同一循环,
只差"是否有人顾问"。

### 5.3 与 src 共享核心的依赖映射

| nemotron 使用 | src 提供 | 说明 |
|---|---|---|
| 能量打分契约 | `EnergyModel.score_conditioned / score_visible_logits(num_lowest) / sample_hidden_state / discretize_visible_state` | `ContextualEnergyModel` 继承并重写;SA 采样作为常量上下文,梯度走可见单元 |
| Ising 本体 | `BoltzmannMachine`(full_bm) | 哈密顿量 + `condition_sample`;参数随训练更新 |
| 推理宿主 | `QDiffusion` + `QDiffusionConfig` + `SequenceTokenSpec` | `build_nemotron_qdiffusion` 组装;`energy()` 保持三参,上下文直达模型 |
| 无 | `objective()` / `generate()` / `_sample*` | dplm 专用:nemotron 训练循环与解码由原生 generate + hook 承担 |

### 5.4 快速复现链

```bash
pip install -e . && pip install -r example/qdiffusion/nemotron/requirements.txt
# Kaiwu SA 求解需运行许可,按团队既有流程配置(首次求解时交互输入亦可)

python -m example.qdiffusion.nemotron.prepare_pairs \
  --model /path/to/Nemotron-Labs-Diffusion-8B \
  --dataset-jsonl /path/to/private_math.jsonl \
  --split train --output-dir /path/to/pairs_train --device cuda:0
python -m example.qdiffusion.nemotron.merge_pairs \
  --inputs /path/to/pairs_train/pairs.pt /path/to/pairs_val/pairs.pt \
  --output /path/to/pairs_merged.pt
python -m example.qdiffusion.nemotron.train \
  --pairs /path/to/pairs_merged.pt \
  --output-dir /path/to/run --device cuda:0
python -m example.qdiffusion.nemotron.evaluate \
  --model /path/to/Nemotron-Labs-Diffusion-8B \
  --checkpoint /path/to/run/best.pt \
  --dataset-jsonl /path/to/private_math_eval.jsonl \
  --output-dir /path/to/eval_output --device cuda:0 \
  --strategies native bm
```
