# 分子设计复现：CIM、SQA 与 SA

本示例实现“分子图指纹 → 条件受限玻尔兹曼机（CRBM）→ QUBO → 候选分子”的生成流程，比较真实 CIM、软件模拟量子退火 SQA 与经典模拟退火 SA。

本目录包含 59 个 Python 文件和本 README，保留复现代码说明与已完成实验的结果摘要。配置、模型权重、输入数据及逐步结果需要另行准备，不随本次源码提交附带。

方法参考 Ajagekar & You, [Molecular design with automated quantum computing-based deep learning and optimization](https://www.nature.com/articles/s41524-023-01099-0), npj Computational Materials 9, 143 (2023)。本实验复用同一个 format-v3 checkpoint；模型训练采用经典采样代理，CIM 用于分子生成阶段。

## 代码说明

代码在示例目录内按用途组织：

```text
molecular_design/
├── README.md
├── cim_sa/                 # CIM／SA 生成入口与 Python 包
├── sqa/                    # SQA 生成入口与 Python 包
├── scripts/                # 运行与结果核验
└── tests/                  # 方法测试
```

| 目录／文件 | 用途 |
|---|---|
| `cim_sa/` | CIM／SA 生成实现，含图指纹、基于 KPP 的 CRBM、结构 QUBO、14-bit 精度处理与候选选择 |
| `sqa/` | 共用同样的 KPP 模型适配，增加显式横场采样及逐步候选预算匹配 |
| `scripts/run_sqa_cim_matched.py` | SQA 并行运行、缓存与恢复 |
| `scripts/verify_results.py` | 从逐步选择记录重算并核验结果表，无求解器依赖 |
| `tests/test_sqa_contracts.py` | 能量符号、gauge 编码、软件 SQA 采样与预算／缓存测试 |
| `tests/test_kpp_crbm.py` | 真实 KPP 模型、旧权重加载、能量／梯度、条件边缘化、自旋解码及训练测试 |

源码中的 `data/`、`models/`、`optimization/`、`evaluation/` 分别负责分子特征、模型、采样生成与指标计算。CIM／SA 使用 `cim_sa/run_generation_strict.py` 的 `--solver kaiwu_cim`／`--solver kaiwu`；SQA 控制器调用 `sqa/run_generation_strict.py`。两组代码各自保留 `qc_mol_design/` 包名和内部导入结构，运行和测试时使用独立进程。

入口、控制器与测试的路径已适配上述目录。完整运行所需的外部文件及其放置位置见下文。

### KPP RBM／BM 复用

`models/crbm.py` 直接导入 `kaiwu.torch_plugin.RestrictedBoltzmannMachine` 和 `BoltzmannMachine`。本次适配核对的上游版本为 [KPP dca313e](https://github.com/qboson/kaiwu-pytorch-plugin/tree/dca313e29c4f6b0b96f406cee3c919eea1114973/src/kaiwu/torch_plugin)。

| 位置 | 复用方式 |
|---|---|
| `CRBM.rbm` | 由 KPP RBM 管理可见／隐藏偏置与耦合参数，`CRBM.energy()` 调用其 `forward()` 后加入指纹条件项 |
| CRBM 条件扩展 | 保留指纹映射 `W1/W2`、按属性组的 categorical 采样、自由能、MAP 标签与目标边缘化；这些功能需要在基础 RBM 之上实现 |
| `CRBM.conditioned_bm()` | 固定一个指纹，把条件偏置和 one-hot 惩罚组成 KPP BM；惩罚引入可见节点之间的耦合，因此使用完整 BM |
| `training/qc_trainer.py` | 调用 KPP `BM.sample()` 完成 Ising 转换和辅助自旋解码，以 Neal 适配器执行本地 SA；保留重复 reads 和各样本的条件索引 |
| `CRBM.build_neg_phase_qubo()` | 从 KPP BM 导出上三角 QUBO，供旧接口使用 |

旧 checkpoint 中的 `b/c/W` 加载时自动映射为 `rbm.linear_bias` 和 `rbm.quadratic_coef`，`W1/W2` 原样保留；新保存的权重使用 KPP 参数名。CD 的更新继续作用于这些底层参数。分子生成阶段仍使用原来的结构 QUBO、14-bit 编码和 CIM／SQA／SA 求解器。

QC 训练从直接 QUBO 采样改为 KPP BM 的 Ising 采样，二者能量等价，但辅助自旋会改变随机退火轨迹。使用新训练代码得到的模型需要重新评估。下方结果表来自接入 KPP **之前**已完成的实验，本次适配没有重跑正式三方法实验。

## 实验设置

- 25 个参考分子 × QED／LogP 两种性质 × 每种 5 个区间，共 250 条轨迹；每条 10 步探索 + 90 步优化，基础种子 42。
- 三方法使用相同 checkpoint、参考清单、目标区间、14-bit 编码和惩罚系数规则；先选择有效候选，再按原 QUBO 能量排序，实际命中标签不参与选择。
- CIM 使用 Kaiwu `CIMOptimizer`，优化模式 `quota`，每次返回 10 个候选。
- SQA 使用 `PathIntegralAnnealingSampler`，连续时间路径积分、1000 sweeps；横场从 2 几何衰减至 0.001，最后设为 0。
- SA 使用 Kaiwu `SimulatedAnnealingOptimizer`，1000 sweeps，每步 5 次 × 10 个候选。

## 实验结果

每步选择一个结果，统计包含无效步骤、不去重。**有效率**为有效分子数／总步数；**命中率**为有效且目标性质位于指定区间的结果数／总步数，区间为左闭右开。

| 方法 | 生成步数 | 有效数／有效率 | 命中数／命中率 | QED 命中率 | LogP 命中率 | 候选总数 |
|---|---:|---:|---:|---:|---:|---:|
| CIM | 25,000 | 24,696／98.78% | 1,511／6.04% | 5.14% | 6.95% | 557,830 |
| SQA | 25,000 | 24,997／99.99% | 1,387／5.55% | 4.73% | 6.37% | 557,830 |
| SA | 25,000 | 25,000／100.00% | 1,398／5.59% | 4.90% | 6.28% | 1,250,000 |

在本次实验中，CIM 的总体命中率最高，比 SQA 多命中 124 个结果；SQA 的有效率比 CIM 高约 1.20 个百分点。

CIM 全量结果包含多次采样与单次续跑两部分。按同一条件和步骤匹配其中 17,318 个单次采样步骤，结果为：

| 方法 | 有效率 | 命中数／命中率 | 每步候选数 |
|---|---:|---:|---:|
| CIM | 98.24% | 852／4.92% | 10 |
| SQA | 99.98% | 826／4.77% | 10 |
| SA | 100.00% | 804／4.64% | 50 |

CIM 与 SQA 的逐步候选数一致；SA 使用更高预算。SQA 预先采用 CIM 已观测预算，但各自独立更新代理和后续矩阵，额外补采样的决策规则也有差别。单次阶段继承此前的轨迹状态。部分参考分子用于前期 CIM 调参。结果基于一个种子，属于描述性比较，不构成统计显著性或硬件加速证明。

## 核验与运行

CSV 数据加载器会将 `qed`、`logp`、`sas` 转换为数值，并丢弃含非数字、NaN 或
正负无穷属性的行；预计算属性与现场计算属性均采用这项检查。合法的数值字符串仍可使用。

运行命令前先进入本示例目录，并按用途准备以下外部文件；这些文件不包含在本次提交中：

| 用途 | 需要准备的文件／目录 |
|---|---|
| CIM／SA／SQA 分子生成 | `zinc250k.csv`、`references_25.json`、`checkpoints/checkpoint_qc.pt`（format-v3，九种重原子特征） |
| 通过 SQA 控制器匹配原实验预算 | 上述生成输入，以及 `configs/sqa.json`、`configs/budgets/*.json`（250 条轨迹预算）、`inputs.json`（输入路径与 SHA-256） |
| 核验原实验结果 | `results/comparison.csv`、`results/selected_steps.csv.gz`、`results/provenance.json` |

取得原始结果文件并按上述路径放置后，仅需 Python 3.10+ 即可核验：

```bash
python scripts/verify_results.py
```

核验包含三方法各 25,000 步、CIM／SQA 的逐步候选数、命中标签及全量／分段统计；不执行采样或重新计算分子性质。

原实验版本为 Python 3.10.20、PyTorch 2.6.0、RDKit 2026.3.2、Kaiwu 1.0.33a0、dwave-samplers 1.8.0。本次在该环境中加载上述真实 KPP 源码，验证模型与本地采样接口。KPP 当前声明的依赖为 PyTorch 2.7.0、NumPy 2.2.6、Kaiwu 1.3.1；本次未验证该完整依赖组合及其 CIM 硬件接口。

在 KPP 仓库的 `example/molecular_design/` 目录内，按仓库依赖安装插件及示例依赖，然后运行方法测试：

```bash
conda create -n qc-mol-reproduction python=3.10.20 pip
conda activate qc-mol-reproduction
python -m pip install ../..
python -m pip install pandas scipy scikit-learn matplotlib tqdm rdkit \
  dwave-samplers==1.8.0 dimod==0.12.22 dwave-neal==0.6.0
KPP_STATS_ENABLED=false python -m unittest discover -s tests -v
QC_MODEL_SOURCE=cim_sa KPP_STATS_ENABLED=false \
  python -m unittest discover -s tests -p test_kpp_crbm.py -v
```

KPP 测试默认使用 `sqa` 源码，第二条测试命令单独验证 `cim_sa`；每套包含 8 项测试，另有 3 项 SQA 测试。CUDA 检查在无 GPU 时跳过。上述测试不初始化 Kaiwu 授权、不调用 CIM，也不需要原实验数据或模型。

下列完整生成命令需要先准备表中对应的外部输入；SQA 控制器会按外部 `inputs.json` 检查数据、模型及参考分子的哈希。准备完成后，使用 SQA 重新生成全部轨迹：

```bash
python scripts/run_sqa_cim_matched.py \
  --scope full --workers 8 \
  --output-root results/new_sqa
```

`--workers` 按可用 CPU 数调整；当前并行控制器使用 Linux 的 fork／文件锁。上述命令生成新的 SQA 轨迹，输出到 `results/new_sqa`。CIM／SA 的生成入口位于各自源码目录中，单次新运行不能重建已经完成的混合预算 CIM 轨迹。

单个 SA 条件的生成示例（100 步、每步 50 个候选）：

```bash
CUDA_VISIBLE_DEVICES="" QC_SOLVER_REPEATS=5 QC_MAX_FEASIBILITY_BATCHES=1 \
QC_CANDIDATE_SELECTION=valid_energy QC_SPIN_ENCODING=gauge \
python cim_sa/run_generation_strict.py \
  --solver kaiwu --prop qed --reference-ids R00 --bands 0 \
  --qubo-mode strict --lambda-mode cim_precision --cim-penalty-ratio 4 \
  --reads 10 --sweeps 1000 --n-explore 10 --n-optimize 90 \
  --run-seed 42 --save-dir results/new_sa/qed_b0_R00_s42
```

SA 需预先激活 Kaiwu 授权，或设置 `KAIWU_USER_ID` 和 `KAIWU_SDK_CODE`。要进行新的单次采样 CIM 运行，将上例改为 `--solver kaiwu_cim`、`QC_SOLVER_REPEATS=1`，换用新的输出目录，并设置：

| 环境变量 | 内容 |
|---|---|
| `KAIWU_CREDENTIALS_FILE` | 本地私有 JSON 文件的绝对路径，含 `user_id`、`sdk_code` |
| `KAIWU_CIM_PROJECT_NO` | 自己的 CIM 项目号 |
| `KAIWU_CIM_RUN_ID` | 本次运行的唯一名称；不同条件使用不同名称 |
| `KAIWU_CIM_CACHE_DIR` | 本次运行的本地缓存目录，如 `results/new_cim/cache` |
| `KAIWU_CIM_TASK_MODE` | `quota`（优化模式，也是默认值） |
| `KAIWU_CIM_BIT_WIDTH` | `14`（默认值） |

凭证文件通过本地私有路径提供。CIM 命令会向自己的项目提交硬件任务；该命令用于新的单次采样轨迹，历史混合预算结果见上方实验结果表。
