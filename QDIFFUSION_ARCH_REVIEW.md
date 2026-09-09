# qdiffusion 架构评审（2026-09-07）

> **临时文档约定**：本文件是一次性评审工作记录，**全部条目处理完毕后删除**。
> 逐项完成时在文末清单勾选；清单全勾 = 删除本文件。

范围：`src/kaiwu/torch_plugin/qdiffusion.py`、`_qdiffusion_sampling.py`、
`example/qdiffusion/`（nemotron / dplm / simple）整体作为一个仓库审视。
状态：**仅记录，未改动**。每项含问题描述、证据（file:line）、影响、修复方向。

健康面（保留优点，不再展开）：EnergyModel 基类 + 采样器依赖注入；版本化
checkpoint + 加载校验；fingerprint 断点续跑；pair schema 校验。

---

## 高优先级

### H1. `QDiffusion.energy()` 门面与上下文能量模型之间没有协议

- **现象**：nemotron 的 `ContextualEnergyHook` 绕过门面直插内层模型打分，
  `guidance.py:130` 直接调 `self.qdiffusion.energy_model.score_conditioned(...)`
  并附注释解释为何要绕。门面签名固定三参数（`src/.../qdiffusion.py:451`），
  没有 context 传递通道。
- **连带影响**：`QDiffusion.objective()` / `_score_candidates()` 同样无法供给
  上下文 → 上下文能量模型在核心训练/解码路径完全不可用，nemotron 被迫在
  example 里另写训练循环（`train.py` 的 `pair_loss`）。
- **背景**：2026-09-02 拍板的 B 方案记录为"门面条件转发 context"，但当前
  代码门面无 context 参数——决策与实现出现漂移，且无决策记录。
- **方向**：门面加 `**context` 关键字转发（保持基类 `score_conditioned`
  三参数不变，无上下文模型收到 context 时 TypeError 响亮报错——符合 B 方案
  初衷）；`objective`/`_score_candidates` 经 `batch["energy_context"]` 透传；
  补一个带 context 的 dummy EnergyModel 测试守护契约。

### H2. `QDiffusion` 身份过载（一个类四种角色）

- **现象**：同时承担 ①训练 objective 容器 ②重参数解码循环 ③能量打分门面
  ④token/config 载体。nemotron 只需 ③④，却要把 8B 冻结模型当
  `proposal_model` 塞进构造函数（`guidance.py:238-250`
  `build_nemotron_qdiffusion`），docstring 自我声明"nemotron 不要用本类的
  通用生成方法"。
- **影响**：API 面与实际使用面错位；构造重（freeze/eval 机制全跑一遍）；
  H1 的绕行由此加剧。
- **方向**：拆轻量 `EnergyScorer`（token_spec + energy_model + 采样参数），
  `QDiffusion` 组合它；`ContextualEnergyHook` 持 scorer。与 H1 配合落地。

### H3. 吞吐架构：SA 求解在打分内环 + 全链 batch=1

- **现象**：打分路径 `score_conditioned → score_visible_logits →
  sample_hidden_state`（`models/energy.py:265` → src）逐行 Python 循环、
  每行一次 `condition_sample`（一次 SA solve）。hook 每步把 K 个候选
  expand 成 batch（`guidance.py:118-119`），K=4 → 每解码步 4 次求解器调用。
  全链写死单样本：`prompt_ids` 返回 `[1, seq]`、`evaluate.py` 单题循环、
  candidates/proposal 里 `squeeze(0)` 十余处。
- **影响**：作为展示 CIM/量子加速能力的示例，最该加速的位置被架构钉死。
- **方向**（短期不动 RNG 语义）：同 state 同候选的 BM 解缓存/去重；
  `evaluate` 层面题间并行；长期做候选维批量求解。

### H4. 依赖可安装性：双 requirements 互斥 + kaiwu pin 失效

- **现象**：`example/qdiffusion/requirements.txt` 钉 `transformers==4.39.2`
  （dplm/esm），`nemotron/requirements.txt` 要求 `transformers>=5.0,<6`，
  同环境装两个文件必坏一个。pyproject 钉 `kaiwu==1.3.1`，PyPI 该版本只有
  cp310 轮子（2026-09-07 实测），py3.12 下 `pip install -e .` 失败；实际
  开发环境跑本地 editable enterprise 1.4.0。
- **方向**：两示例改互斥 extras + 分环境安装文档；kaiwu 放宽为
  `>=1.3.1` 或加 `python_version` marker。

---

## 中优先级

### M1. 设备/精度缓存漏风

只 override `to()`/`train()`（`qdiffusion.py:403,420`）；`.cuda()`/`.bfloat16()`/
`.half()` 走 `_apply()` 不经过 `to()` → `self.device`/`self.dtype` 陈旧。
`models/energy.py:154-158` 的 BM device 手工同步 TODO 同根因。
**方向**：override `_apply` 而非 `to`。

### M2. 魔法字符串与弱类型边界

`decoding_strategy` 在解码中途 `split("-")` 逐段解析（`qdiffusion.py` 987 附近），
配置错误延迟到生成第 N 步才报错，且须恰好 4 段。解码 state 是
`dict[str, Any]`。`QDiffusionConfig` 按引用共享（`config or QDiffusionConfig()`）。
**方向**：strategy 配置期校验或策略对象；state 换 dataclass/TypedDict；
config 存副本。

### M3. hook 与 QDiffusion 参数双轨

hook 自收 `num_candidates`/`proposal_temperature`/`proposal_noise_scale`
（`guidance.py:44-46`），`build_nemotron_qdiffusion` 又把同名参数塞进
`QDiffusionConfig`（`guidance.py:242-246`），nemotron 流程里后者从未被消费。
`load_nemotron_guidance` 返回的 qdiffusion 在 `evaluate.py` 唯一用途是
`assert not None`。hook 直取 `proposal_model.get_input_embeddings()`（HF 约定）。
**方向**：随 H2 拆分自然消解；参数单轨化。

### M4. 版本化散落

`PAIR_SCHEMA_VERSION`（pairs.py）、`COLLECTOR_SCHEMA_VERSION`
（prepare_pairs.py）、`CHECKPOINT_FORMAT`（energy.py）三处独立维护；
`train.py` 手工拼 `_compatibility_config` 字典。
**方向**：集中 schema/compat 模块，单一事实源。

### M5. CLI 参数面重复

`model/device/seed/block_length/threshold/max_new_tokens` 在 prepare_pairs 与
evaluate 各自声明，已漂移过一次（`--K`）。
**方向**：common/ 提供共享 parser 片段。

### M6. `_sample_candidates` 布局假设写死 / stats 边通道

布局假定 `[n, batch, seq]`（旧三布局适配器已删），采样器契约无守护；
`_last_stats` 为"打分时写、按顺序读"的边通道，顺序耦合、非线程安全。
**方向**：采样器契约测试；stats 改显式返回值或上下文对象。

---

## 低优先级

### L1. 示例分发故事缺失

dplm_builder `try/except ImportError` 双导入路径（脚本模式兜底，需同时插
workflows/ 与 dplm/ 两个目录）；simple 示例依赖 repo root 在 sys.path。

### L2. `_resample`/`score_visible_logits` Python 循环、逐行 solver.solve

已明确推迟（改并行会动 RNG 消耗顺序、破坏可复现性）——与 H3 相关但独立跟踪。

### L3. README 维护双份成本

README.md 与 README_ZH.md 内容已对等，后续改动需双写；可考虑单源生成。

---

## 测试策略盲区

"example 代码不进 tests/"把最该测的纯逻辑留在无守护区：
`build_diverse_transfer_candidates` 去重回填、`ForcedCandidateHook` 回放分歧
检测、pair 校验规则、fingerprint 续跑比对——均为不需要 GPU/模型权重的纯函数。
**方向**：策略改为"依赖模型权重的测试不进"，纯逻辑模块进 `tests/`。

---

## 建议动手顺序

- [ ] 1. H1（门面 context 协议，小改动、恢复既定决策、可带契约测试）
- [ ] 2. H4（依赖可安装性，现在就在咬人）
- [ ] 3. M1（`_apply` override，低成本真 bug）
- [ ] 4. 纯逻辑测试准入
- [ ] 5. H2 + M3（EnergyScorer 拆分，结构性重构，单独立项）
- [ ] 6. H3 / L2（吞吐，单独立项）
