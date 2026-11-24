# 语言版本：[中文](README_ZH.md) | [English](README.md)

# `example/gbrbm_anomaly_detection`

本目录给出一个基于高斯-伯努利受限玻尔兹曼机（Gaussian-Bernoulli RBM，GBRBM）的
**连续特征无监督异常检测**端到端案例：只用正常样本训练，再用模型的自由能与确定性
重构误差给样本打分，并在同一份留出集上与经典 `IsolationForest` 逐项对比。

## 案例概述

- **零核心改动**：不修改 `src/kaiwu/torch_plugin/` 下任何文件，只使用
  `GaussianBernoulliRestrictedBoltzmannMachine` 的公开接口
  （`marginal_energy`、`infer_from_gaussian`、`infer_from_bernoulli`、`sample`、`objective`）。
- **严格防泄漏**：先按原始样本划分 `train_normal` / `val` / `test`，再**只在正常训练集**上
  拟合 `StandardScaler`；验证集与测试集的分布和极值完全不参与均值/方差计算。
- **两种评分方向一致**：自由能与重构误差都满足“分数越高越异常”，可直接送入
  `roc_auc_score` / `average_precision_score`。
- **默认本地采样**：默认使用 `SimulatedAnnealingOptimizer`，不提交任何云端任务；
  另提供 `--sampler cim` 入口并兼容 Kaiwu SDK 1.3.1–1.4.3。
- **非阻塞绘图**：绘图模块只使用 `Figure` + `FigureCanvasAgg`，不创建 GUI 窗口、
  不调用 `plt.show()`；未安装 `matplotlib` 时主流程自动降级。

## 目录结构

```text
example/gbrbm_anomaly_detection/
├── ad_config.py               # frozen dataclass 配置与结果结构
├── ad_data.py                 # 数据加载、无泄漏划分与标准化
├── ad_sampler.py              # 本地 SA 默认构建、CIM 兼容入口、授权环境变量
├── ad_scorer.py               # GBRBMAnomalyScorer（自由能 / 重构误差）
├── ad_trainer.py              # 无监督训练循环、设备与随机种子管理
├── ad_eval.py                 # 验证集阈值选取、测试集指标、基线与 JSON 导出
├── ad_plot.py                 # 纯 OO Agg 画布，非阻塞保存 PNG
├── run_anomaly_detection.py   # CLI 入口与端到端编排
├── requirements.txt           # 案例增量依赖
└── README_ZH.md / README.md   # 中英文说明
```

## 数据集与防泄漏划分

数据来自 `sklearn.datasets.load_breast_cancer()`（Breast Cancer Wisconsin Diagnostic），
569 条样本、30 维连续细胞核形态特征，随 `scikit-learn` 本地分发，无需下载。

标签定义：**良性（`target == 1`，357 条）记为正常类 `0`；恶性（`target == 0`，212 条）
记为异常类 `1`**。

| 划分 | 样本数 | 正常 | 异常 | 用途 |
| --- | ---: | ---: | ---: | --- |
| `train_normal` | 214 | 214 | 0 | GBRBM 与 `IsolationForest` 的无监督训练 |
| `val` | 177 | 71 | 106 | 选取判定阈值 |
| `test` | 178 | 72 | 106 | 仅用于最终报告 |

划分流程（`ad_data.py`）：

1. 取出全部正常索引，按 `train_normal_ratio=0.6` 切出纯正常训练集与留出正常样本；
2. 把留出正常样本与全部异常样本合并，按 `stratify` 以 `val_ratio_of_eval=0.5` 分层切分为
   `val` 与 `test`，保证两者异常比例接近；
3. 分割完成后才创建 `StandardScaler`，并**只** `fit(features[train_indices])`；
4. 同一个 scaler 变换三个子集，统一转为 `float32`。

所有索引均保留在 `AnomalyDatasetSplit.train_indices / val_indices / test_indices` 中，
便于单元测试直接断言三者两两不相交且并集为全体样本。

## 异常评分方法

`GBRBMAnomalyScorer`（`ad_scorer.py`）只调用现有公开接口，并在内部把输入对齐到
`(model.device, model.dtype)`。记连续输入 $v \in \mathbb{R}^{D}$、隐层 $h \in \{0,1\}^{H}$、
均值 $\mu$、方差 $\sigma^2 = \exp(\text{log\_var})$、权重 $W$、隐层偏置 $b$。

### 1. 自由能（默认主评分）

```text
F(v) = 0.5 * Σ_i (v_i - μ_i)² / σ_i²  -  Σ_j softplus( Σ_i v_i W_ij / σ_i² + b_j )
```

`marginal_energy` 在 `torch.no_grad()` 下解析求和消去隐层，得到精确自由能。
由于 $p(v) \propto \exp(-F(v))$，低概率的异常样本天然获得更高的 $F(v)$，
**无需取负号**，且当 $\lVert v \rVert \to \infty$ 时二次项主导使 $F(v) \to +\infty$。

### 2. 确定性重构误差（辅助评分）

```text
p(h=1|v) = sigmoid( v W / σ² + b )        # infer_from_gaussian(binarize=False)
v̂        = p(h=1|v) Wᵀ + μ                # infer_from_bernoulli(no_random=True)
s_RE(v)  = (1/D) * Σ_i (v_i - v̂_i)²
```

编码保留条件激活概率、解码使用条件均值，因此该评分在测试期**没有采样噪声**，
可重复性最好；重构误差越大，样本越偏离正常流形。

两个方法的符号约定一致：**分数越高越异常**。

## 训练流程

`ad_trainer.train_gbrbm()` 只接收纯正常训练集：

1. `resolve_device()` 解析 `auto` / `cpu` / `cuda`（请求 `cuda` 但不可用时显式报错，不静默回退）；
2. `set_global_seed()` 同步固定 Python `random`、`np.random` 与 `torch.manual_seed`；
3. 构造 `GaussianBernoulliRestrictedBoltzmannMachine(..., dtype=torch.float32, device=...)`，
   并在 `freeze_log_var=True`（默认）时执行 `model.log_var.requires_grad_(False)`；
4. 每个 mini-batch 执行对比散度式更新：
   - 正相 `s_pos = infer_from_gaussian(batch, binarize=False)`；
   - 负相 `s_neg = model.sample(sampler)`（由采样器求解 Bernoulli 侧 Ising 模型）；
   - 目标 `loss = objective(s_pos, s_neg) + weight_decay * (‖W‖² + ‖b‖²)`；
   - 反向传播后按 `grad_clip_norm` 裁剪梯度并更新。

**为什么默认冻结 `log_var`**：`GBRBM.sample()` 内部固定使用
`infer_from_bernoulli(..., no_random=True)`，负相高斯态取条件均值而未叠加
$\mathcal{N}(0, \sigma^2)$ 噪声。对 $\mu, W, b$ 而言这是无偏的
Rao-Blackwellized 估计；但对 `log_var`，真实期望
$\mathbb{E}_{v|h}[(v_i-\mu_i)^2] = ((Wh)_i)^2 + \sigma_i^2$ 中缺失的 $+\sigma_i^2$
会使梯度产生系统性偏差，进而诱发方差漂移。因此在标准化空间（零均值、单位方差）内
按 Hinton (2010) 的经典实践保持 $\sigma_i^2 = 1$，并通过
`ModelConfig.freeze_log_var` 保留显式开关。

## 采样器：本地 SA（默认）与 CIM 入口

`ad_sampler.build_sampler()`：

| `--sampler` | 行为 |
| --- | --- |
| `sa`（默认） | `kaiwu.classical.SimulatedAnnealingOptimizer`，本地运行，不触发云端任务 |
| `cim` | `kaiwu.cim.CIMOptimizer`，默认外包 `PrecisionReducer`，需要已授权的 Kaiwu 账号 |

CIM 入口通过 `kaiwu.torch_plugin._kaiwu_compat` 的 `normalize_task_mode()` 与
`PrecisionReducer` 统一 Kaiwu 1.3.1 与 1.4.0–1.4.3 的差异：

| `--cim-task-mode` | Kaiwu 1.3.1 | Kaiwu 1.4.x |
| --- | --- | --- |
| `sampling` / `sample` | `"sample"` | `TaskMode.SAMPLING` |
| `optimization` / `quota` | `"quota"` | `TaskMode.OPTIMIZATION` |

### 授权配置（严禁硬编码密钥）

`ad_sampler.init_kaiwu_license_from_env()` 按优先级读取成对的环境变量，仅当两者均非空时
调用 `kw.license.init(user_id, sdk_code)`：

1. `USER_ID` + `SDK_CODE`
2. `LICENSE_USER_ID` + `LICENSE_SDK_CODE`

```bash
# Linux / macOS
export USER_ID="<your user id>"
export SDK_CODE="<your sdk code>"
```

```powershell
# Windows PowerShell
$env:USER_ID = "<your user id>"
$env:SDK_CODE = "<your sdk code>"
```

未配置凭据时脚本不会伪造或绕过授权，只保持 SDK 当前授权状态；本地 SA 的退火求解仍由
Kaiwu SDK 自身的授权校验决定。请勿把密钥写入代码、示例配置或提交到仓库。

## 快速开始

```bash
pip install -e .                                       # 仓库根目录安装插件
pip install -r example/gbrbm_anomaly_detection/requirements.txt

# 数秒级冒烟：2 epoch、8 个隐单元、更轻的 SA 计划
python example/gbrbm_anomaly_detection/run_anomaly_detection.py --quick --seed 42 --device cpu

# 完整参考配置：30 epoch、16 个隐单元
python example/gbrbm_anomaly_detection/run_anomaly_detection.py --seed 42 --device cpu \
    --output-dir example/gbrbm_anomaly_detection/outputs/full

# 跳过绘图（CI 无 matplotlib 时也会自动跳过）
python example/gbrbm_anomaly_detection/run_anomaly_detection.py --quick --no-plot
```

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--epochs` | int | `30` | 训练轮数；`--quick` 时强制为 `2` |
| `--seed` | int | `42` | 数据划分、PyTorch、SA 与 `IsolationForest` 的全局种子 |
| `--output-dir` | path | `example/gbrbm_anomaly_detection/outputs` | `metrics.json` 与两张 PNG 的输出目录 |
| `--device` | str | `auto` | `auto` / `cpu` / `cuda`（`cuda` 不可用时显式报错） |
| `--quick` | flag | 关闭 | 快速模式：`epochs=2`、`num_hidden=8`、`sa_size_limit=16`、`sa_iterations_per_t=3`、`sa_alpha=0.9` |
| `--sampler` | str | `sa` | `sa`（本地模拟退火）/ `cim`（相干光量子真机入口） |
| `--score-method` | str | `free_energy` | 直方图使用的评分方式：`free_energy` / `reconstruction`（两者指标都会写入 JSON） |
| `--num-hidden` | int | `16` | 伯努利隐单元数 |
| `--batch-size` | int | `32` | 批大小 |
| `--lr` | float | `5e-3` | SGD 学习率 |
| `--weight-decay` | float | `1e-2` | L2 正则系数 |
| `--no-plot` | flag | 关闭 | 只写 `metrics.json`，不生成图片 |
| `--cim-task-name` | str | `gbrbm_anomaly_detection` | CIM 任务名（仅 `--sampler cim`） |
| `--cim-task-mode` | str | `sampling` | `sampling` / `sample` / `optimization` / `quota` |
| `--cim-no-precision-reducer` | flag | 关闭 | 关闭 CIM 的 `PrecisionReducer` 包装 |

## 输出产物

```text
<output-dir>/
├── metrics.json              # 配置、划分统计、训练历史与三组完整指标
├── score_distribution.png    # 测试集正常/异常评分直方图 + 验证集阈值虚线
└── roc_curve_comparison.png  # GBRBM（自由能 / 重构）与 IsolationForest 的 ROC 对比
```

`metrics.json` 的 `metrics` 字段包含 `gbrbm_free_energy`、`gbrbm_reconstruction`、
`isolation_forest` 三组 `roc_auc / pr_auc / f1 / precision / recall / val_f1 / threshold`。

## 参考结果

完整配置（`--seed 42 --device cpu`，30 epoch，16 个隐单元），测试集 178 条（106 条异常），
阈值全部由验证集选定后原样用于测试集：

| 模型 / 评分 | ROC-AUC | PR-AUC | F1 | Precision | Recall | 验证集阈值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GBRBM（自由能） | 0.9434 | 0.9622 | 0.8789 | 0.8376 | 0.9245 | 10.9980 |
| GBRBM（重构误差） | 0.9434 | 0.9620 | 0.8829 | 0.8448 | 0.9245 | 1.4692 |
| `IsolationForest` | 0.9450 | 0.9593 | 0.8826 | 0.8785 | 0.8868 | -0.0043 |

训练过程：平均目标函数 `15.0235 → 14.5062`，正常训练集平均自由能 `3.9229 → 3.5606`。

`--quick` 冒烟模式（2 epoch、8 个隐单元）同样能跑通全链路，自由能评分给出
ROC-AUC `0.9441` / PR-AUC `0.9626` / F1 `0.8778`。

需要说明的是：SA 与 CIM 都是随机采样器，Ising 求解质量受温度计划、`size_limit` 与
SDK 版本影响，因此上述数值在不同运行、不同 Kaiwu 版本与不同硬件上会有小幅波动；
`IsolationForest` 在 `random_state=42` 下可完全复现。

## 第三方依赖与开源协议

本案例不引入新的重型依赖，全部已在仓库既有依赖与 `NOTICE` 声明范围内，并符合 CI 的
`pip-licenses --allow-only "MIT;LGPL;BSD;MPL;Apache;Unlicense;PSF"` 白名单。

| 依赖 / 数据 | 用途 | 协议 |
| --- | --- | --- |
| `torch` | GBRBM 张量运算与训练/推理 | BSD-3-Clause |
| `numpy` | 数值数组与指标前处理 | BSD-3-Clause |
| `kaiwu` | `SimulatedAnnealingOptimizer` / `CIMOptimizer` / `PrecisionReducer` | Kaiwu SDK 专有许可（插件既有依赖） |
| `scikit-learn` | 数据集、划分、`StandardScaler`、`IsolationForest`、ROC/PR/F1 | BSD-3-Clause |
| `matplotlib`（可选） | `FigureCanvasAgg` 保存 PNG | PSF-based（Matplotlib License） |
| Breast Cancer Wisconsin (Diagnostic) | 569 条 30 维连续特征基准数据 | CC BY 4.0（随 `scikit-learn` 内置分发） |
