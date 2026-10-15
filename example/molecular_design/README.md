# 分子设计复现：CIM、SQA 与 SA

本示例实现“分子图指纹 → 条件受限玻尔兹曼机（CRBM）→ QUBO → 候选分子”的生成流程，比较真实 CIM、软件模拟量子退火 SQA 与经典模拟退火 SA。

方法参考 Ajagekar & You, [Molecular design with automated quantum computing-based deep learning and optimization](https://www.nature.com/articles/s41524-023-01099-0), npj Computational Materials 9, 143 (2023)。本实验复用同一个 format-v3 checkpoint；模型训练采用经典采样代理，CIM 用于分子生成阶段。

## 代码说明

代码在示例目录内按用途组织：

```text
molecular_design/
├── README.md
├── cim_sa/                 # CIM／SA 生成入口与 Python 包
├── sqa/                    # SQA 生成入口与 Python 包
├── configs/                # cim_sa.json、sqa.json、budgets/
├── scripts/                # 运行与结果核验
├── tests/                  # 方法测试
├── checkpoints/            # 共用模型
├── results/                # 汇总表、逐步结果与来源记录
├── references_25.json
├── inputs.json
├── environment.reproduction.yml
└── requirements*.txt
```

| 目录／文件 | 用途 |
|---|---|
| `cim_sa/` | CIM／SA 的冻结生成实现，含图指纹、CRBM、结构 QUBO、14-bit 精度处理与候选选择 |
| `sqa/` | SQA 的冻结生成实现，增加显式横场采样及逐步候选预算匹配 |
| `configs/budgets/` | 250 条轨迹的固定采样预算 |
| `scripts/run_sqa_cim_matched.py` | SQA 并行运行、缓存与恢复 |
| `scripts/verify_results.py` | 从逐步选择记录重算并核验结果表，无求解器依赖 |
| `references_25.json` | 固定的 25 个参考分子 |
| `checkpoints/checkpoint_qc.pt` | 本次三方法共用模型 |
| `results/` 下的 `comparison.csv`、`selected_steps.csv.gz`、`provenance.json` | 汇总表、75,000 条最终选择记录及来源校验值 |

冻结源码中的 `data/`、`models/`、`optimization/`、`evaluation/` 分别负责分子特征、模型、采样生成与指标计算。CIM／SA 使用 `cim_sa/run_generation_strict.py` 的 `--solver kaiwu_cim`／`--solver kaiwu`；SQA 控制器调用 `sqa/run_generation_strict.py`。两组代码各自保留 `qc_mol_design/` 包名和内部导入结构。

入口、控制器与测试的路径已适配上述目录，模型和求解器实现保留实验版本。`configs/cim_sa.json` 记录原 CIM 续跑配置，历史输出和缓存路径仅用于溯源；`configs/sqa.json` 与预算文件用于新的 SQA 运行。

当前实现直接调用 PyTorch 和 Kaiwu SDK，自定义 CRBM 尚未迁移为 `kaiwu.torch_plugin` 模型类。

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

进入本示例目录，仅需 Python 3.10+ 即可核验已保存结果：

```bash
python scripts/verify_results.py
```

核验包含三方法各 25,000 步、CIM／SQA 的逐步候选数、命中标签及全量／分段统计；不执行采样或重新计算分子性质。

重新运行使用独立环境，安装 `requirements.reproduction.txt` 与 `requirements.solvers.txt`。原实验版本为 Python 3.10.20、PyTorch 2.6.0、RDKit 2026.3.2、Kaiwu 1.0.33a0、dwave-samplers 1.8.0。Kaiwu SDK 单独安装，SA／CIM 使用自己的授权；尚未验证与仓库主环境 Kaiwu 1.3.1 的兼容性。

安装及方法测试：

```bash
conda env create -f environment.reproduction.yml
conda activate qc-mol-reproduction
python -m pip install -r requirements.solvers.txt
python -m unittest discover -s tests -v
```

上述 SQA 方法测试不需要 Kaiwu 授权。生成还需要在本示例目录准备原始 `zinc250k.csv`，其身份校验值见 `inputs.json`。数据 CSV 与完整矩阵／自旋缓存不随代码提交，已有结果核验无需这些文件。使用 SQA 重新生成全部轨迹：

```bash
python scripts/run_sqa_cim_matched.py \
  --scope full --workers 8 \
  --output-root results/new_sqa
```

`--workers` 按可用 CPU 数调整；当前并行控制器使用 Linux 的 fork／文件锁。上述命令生成新的 SQA 轨迹，不覆盖发布结果。CIM／SA 的生成入口在各自冻结源码中，单次新运行不能重建已经完成的混合预算 CIM 轨迹。

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

凭证文件应放在 `.private/` 等已忽略的私有目录。CIM 命令会向自己的项目提交硬件任务；该命令用于新的单次采样轨迹，历史混合预算结果以随附表格为准。
