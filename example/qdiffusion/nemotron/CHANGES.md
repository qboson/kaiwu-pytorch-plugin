# nemotron 文件导览与本分支改动说明

本文档面向 code review:第一部分逐文件说明本目录(全部为新增)的职责,
第二部分汇总本分支对既有文件的改动。方法与命令细节见 [README_ZH.md](README_ZH.md)。

## 一、本目录文件导览(全部新增)

### 入口脚本(仓库根目录 `python -m` 运行)

| 文件 | 职责 |
|---|---|
| `prepare_pairs.py` | 离线采集:重放冻结模型的原生生成,捕获每个 block 决策点的候选分支(`CandidateTraceHook`),在决策最摇摆的点强制走岔路(`ForcedCandidateHook`),按最终答案对错配成正负 pair。逐题原子落盘,支持断点续采 |
| `merge_pairs.py` | 合并多个分片采集的 `pairs.pt`;合并后重新校验 schema 与 train/val split 一致性,能抓到单分片查不出的题目跨 split 泄漏 |
| `train.py` | 训练 `ContextualEnergyModel`:outcome-pairwise margin 为主目标,0.01 权重 NCE 仅作能量尺度正则;按 val ranking 选 `best.pt`;RNG 级精确续跑(恢复 DataLoader 顺序与全部随机状态) |
| `evaluate.py` | Native 与 BM 引导的对照解码评测;boxed-only `math-verify` 判分;每个 strategy 独立 partial 文件与配置指纹,支持 append-only 续评 |

### `common/`(共享地基)

| 文件 | 职责 |
|---|---|
| `runtime.py` | 模型/分词器加载(`trust_remote_code`)、prompt 编码、原生生成共享契约(`temperature=0` 写死贪心)、JSONL 读取、原子写(JSON/pt)、文件 SHA-256 指纹。原 `_common.py` 更名 |
| `answers.py` | 最终答案抽取(嵌套 `\boxed{}` 解析)与 NeMo 兼容的 `math-verify` 等价判定。自 `evaluate.py` 抽出,供采集与评测共用 |
| `pairs.py` | pair 数据层:schema v2、逐条张量形状/transfer-mask 校验、split 泄漏检查、`load/save`。自 `prepare_pairs.py` 抽出 |

### `models/`(能量模型)

| 文件 | 职责 |
|---|---|
| `energy.py` | `ContextualEnergyModel(EnergyModel)`:三个投影器 + Transformer 上下文编码 → KPP BM 512×256 的可见单元 → Kaiwu SA 采样隐藏单元出能量;`visible_transform=identity`(连续可见条件,梯度精确);提供 versioned config 与 compact state 读写。原 `modeling.py` 更名 |
| `checkpoint.py` | 版本化 checkpoint:payload 构造(`checkpoint_format`/归一化统计/`energy_lambda`/训练状态)、原子保存、加载校验 |

### `generation/`(与冻结 LLM 的桥接)

| 文件 | 职责 |
|---|---|
| `proposal.py` | `NativeGenerationSession`(上下文管理器,包装原生 generate 的 cache/去噪/停止)、`ProposalStep`/`ProposalDecision` 快照——hook 机制的地基 |
| `candidates.py` | Gumbel 混合候选生成、proposal logprob 打分、多样性迁移候选构建 |
| `guidance.py` | `ContextualEnergyHook`(在 transfer 决策点调用 `ContextualEnergyModel.score_conditioned` 具名打分,按 residual 规则覆盖/放行)+ `build_nemotron_qdiffusion`/`load_nemotron_guidance` 组装 |

### 其他

| 文件 | 职责 |
|---|---|
| `requirements.txt` | 本目录运行依赖(`math-verify` 等) |
| `CHANGES.md` | 本文档 |
| `QDIFFUSION_ARCH_REVIEW.md` | 一次性架构评审工作记录(约定全部条目处理完毕后删除) |

## 二、本分支对既有文件的改动

### `src/kaiwu/torch_plugin`(发布包)

- `qdiffusion.py`:`score_visible_logits(num_lowest=...)` 支持只取最低 N 个解的能量均值;`energy()` 保持无上下文的三参签名——上下文条件是能量模型的固有能力,由 `ContextualEnergyModel.score_conditioned` 具名接收,nemotron 推理直达模型调用(无上下文模型收到上下文会 TypeError,而非静默忽略);`train()` 保证冻结的 proposal 保持 eval 模式;删除无消费者的 `weight`/`temperature`/`history` API;全函数补齐 Google 风格 docstring
- `__init__.py`:`SequenceTokenSpec` 加入包级导出
- `_qdiffusion_sampling.py`:仅文件头调整
- `abstract/full_boltzmann_machine.py`:本分支无改动(BM 设备修复在独立分支 `fix/bm-device-sync`)

### `tests/`

- `test_energy_model.py`:新增 `score_visible_logits(num_lowest)` 行为测试
- `test_qdiffusion_dummy.py`:新增 `train()` 冻结保护的正反测试;移除 weight 相关断言;测试用 fake 能量模型瘦身为三参签名
- `test_qdiffusion_import.py`:新增 `SequenceTokenSpec` 导出断言

### `example/qdiffusion/dplm`

- 入口收敛:删除 `train_workflow.py`、`eval_esm2_distances.py`、`_example_bootstrap.py`;唯一入口为 `workflows/train.py`(训练)与 `workflows/esm2_eval.py`(评估),`pip install -e .` 后以 `python -m` 运行;内部 import 拍平为包内相对导入
- `models/esm_patch.py`:重写为 SDPA 注意力实现,`forward` 签名精确镜像 4.39.2 父类;裁剪 HF 生成钩子/decoder/位置扩展,保留 contact head 以兼容 strict state_dict 加载
- `utils/runtime.py`:checkpoint 增加 `checkpoint_format` 版本字段,写入改为 tmp+`os.replace` 原子写,加载时校验 payload 类型/版本/必需键
- `workflows/*`:移除手工 `proposal_model.eval()` 拐杖;eval 流程清理(order-only 配对、esm 延迟导入并给出安装提示);补齐 Google 风格 docstring
- `models/{bm,common}.py`:补齐 docstring;`models/backbone.py` 改从包顶层导入 `SequenceTokenSpec`

### `example/qdiffusion/simple`

- 两个最小示例改为通过已安装包导入 `build_qdiffusion`,删除本地路径引导

### 其他

- `example/qdiffusion/README.md` / `README_ZH.md`:入口唯一化说明、`pip install -e .` 前置要求、`python -m` 调用约定
- `example/qdiffusion/requirements.txt`:新增 `fair-esm>=2.0`(ESM2 距离评估的可选依赖)
- 仓库根 `.gitignore`:吸收本目录本地产物模式(`data/`、`runs/`、`checkpoints/`、`*.pt`、`*.jsonl`),子目录不再维护独立 `.gitignore`

## 三、2026-09-07 架构评审批次(已评审,未提交)

本批次由 `QDIFFUSION_ARCH_REVIEW.md` 的评审驱动,共 13 文件 +605/−131。
**性质:文档/命名/风格/一致性维护,无算法行为变更**(但含一处破坏性 CLI 变更,见下)。
评审时修复了批次引入的一处 `train.py:166` 不换行空格(U+00A0)导致的 SyntaxError。

### 逐文件说明

| 文件 | 改动 |
|---|---|
| `common/{answers,pairs,runtime}.py` | 全量补齐 Google 风格 docstring(Args/Returns/Raises),与实现逐条核对一致(含 `read_jsonl` 的 key 校验与空数据集 `ValueError` 声称);`OutcomePair` 补全 15 字段的 `Attributes:` |
| `train.py` / `evaluate.py` / `prepare_pairs.py` | 同上,并为 `_partial_store`/`_append_item`/`ranked_branch_points` 等 resume/hook 方法补齐含 `Raises:` 的完整文档 |
| `generation/candidates.py` | **`K` → `num_candidates` 全链路重命名**(解决 pylint C0103);docstring 补全 |
| `evaluate.py` | **破坏性 CLI 变更:`--K` → `--num-candidates`**;resume 指纹字段 `"K"` → `"num_candidates"`——**旧 partial 文件续跑会报 fingerprint mismatch,需重跑或手工迁移 meta**;docstring 补全 |
| `models/energy.py` | 整除校验注释中译英(语义不变) |
| `README.md` | 英文版补全完整用法(方法边界/依赖/pair schema/三步流程/Notes),消除此前"EN 只有概述"的中英不对称 |
| `simple_generate_example.py` | `decode_tokens` 增加 `QDiffusion` 类型标注(为标注导入该类) |
| `pylintrc` | 注释中译英;补文件末尾换行 |
| `src/.../qdiffusion.py` | docstring 风格收紧:Args 条目间去空行(更贴近 Google 官方紧凑式),类 docstring 措辞调整;**无功能变更** |

### 待决事项(本批次仅记录、未实施)

- 评审 H1:`QDiffusion.energy()` 门面无上下文通道,与 2026-09-02 B 方案(门面条件转发)存在决策漂移;文档建议门面加 `**context` 转发(基类保持三参)。**注意:此方向涉及 kwargs 使用与既有决策,尚待拍板**;本批次未做任何功能性改动,`guidance.py` 仍直达模型调用。
