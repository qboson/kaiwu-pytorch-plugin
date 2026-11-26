# 语言版本：[中文](README_ZH.md) | [English](README.md)

# 基于 RBM 与退火采样的协同过滤推荐

本示例基于 `kaiwu.torch_plugin.RestrictedBoltzmannMachine` 实现了一套模块化、端到端的稀疏协同过滤与 Top-K 推荐案例。模型结合面向观测掩码的对比散度（Masked Contrastive Divergence）与 Ising 退火负相采样（`rbm.sample(sampler)` + `rbm.objective(...)`），并在严格按用户留出（零数据泄漏）的统一评测协议下，与经典 **Popularity** 及 **TruncatedSVD** 基线进行定量对比。

## 算法原理

受限玻尔兹曼机（RBM）在物品可见层 $v \in [0, 1]^V$（归一化评分）与二值隐层潜在偏好因子 $h \in \{0, 1\}^H$ 上定义能量函数：

$$E(v, h) = -a^\top v - b^\top h - v^\top W h$$

针对带观测掩码 $M \in \{0, 1\}^{U \times V}$ 的稀疏用户-物品评分矩阵，本实现包含四个关键设计：

1. **无泄漏训练集先验填充**：在隐层前向推断阶段，未观测条目（$M_{ui} = 0$）使用严格仅由训练集计算的阻尼物品均值 $\mu_i$ 填充，可见层偏置初始化为对数几率 $\log(\mu_i / (1 - \mu_i))$。这既保证不同交互数量用户的隐层输入尺度一致，也能清晰区分“已观测低分（$\tilde{r}_{ui} = 0$）”与“未观测物品（$\mu_i \approx 0.5$）”。
2. **观测掩码对比散度（Masked CD-$k$）**：正相与重构负相能量仅在已观测条目（$M_{ui} = 1$）上聚合，确保未观测物品不会向权重 $W$ 或可见偏置 $a$ 注入伪梯度。
3. **Ising 退火玻尔兹曼正则化**：每轮训练通过 `RestrictedBoltzmannMachine.sample(sampler)` 将 RBM 转为 Ising 矩阵并采样全局玻尔兹曼负相自旋，结合按批次观测密度校准的 `rbm.objective(s_pos, s_anneal)` 参与更新，同时通过 `rbm.clip_parameters(...)` 约束参数动态范围以适配物理 Ising 硬件精度。
4. **Top-K 推荐推断**：基于用户历史观测评分，通过 `rbm.get_hidden` 推断隐层偏好后由 `rbm.get_visible` 重建全量物品预期评分，在严格屏蔽训练集与验证集已见物品后输出 Top-K 推荐列表。

## 目录结构

```text
example/collaborative_filtering/
├── __init__.py       # 包公共接口导出
├── dataset.py        # 确定性合成基准生成器、MovieLens-100K 加载器、按用户留出切分
├── model.py          # RBMCollaborativeFiltering、RBMCFConfig、LocalIsingAnnealingSampler、采样器工厂
├── baselines.py      # PopularityRecommender 与偏置中心化 TruncatedSVDRecommender
├── metrics.py        # RMSE、MAE、Precision@K、Recall@K、NDCG@K（严格排除已见物品）
├── benchmark.py      # 基准调度、CLI 参数解析与对比表格式化
├── train_cf.py       # 可直接运行的命令行入口
├── README.md         # 英文说明文档
└── README_ZH.md      # 中文说明文档
```

## 依赖环境

本示例**仅依赖 `kaiwu-torch-plugin` 核心依赖**（`torch`、`numpy`、`kaiwu`），默认模式离线零下载即可一键运行：

```bash
pip install -r requirements/requirements.txt
pip install .
```

## 快速开始

运行默认离线合成基准（CPU 约 4 秒完成）：

```bash
python example/collaborative_filtering/train_cf.py
```

快速冒烟运行（`--fast`）并导出结构化 JSON 结果：

```bash
python example/collaborative_filtering/train_cf.py --fast --output-json /tmp/cf_results.json
```

运行经典 Gibbs CD-1 消融对比（`--sampler cd`）：

```bash
python example/collaborative_filtering/train_cf.py --sampler cd
```

加载本地 MovieLens-100K（`u.data`）运行（需自行本地下载 `ml-100k` 数据集）：

```bash
python example/collaborative_filtering/train_cf.py \
  --dataset movielens \
  --data-path /path/to/ml-100k/u.data \
  --max-items 100 \
  --num-hidden 24 \
  --epochs 45
```

## 切换至 Kaiwu SDK 模拟退火或相干光量子真机（CIM）

通过 `--sampler` 参数即可在不修改模型代码的前提下无缝切换 Ising 负相采样后端：

| `--sampler` | 采样后端 | 运行前提 |
|---|---|---|
| `local_sa`（默认） | `LocalIsingAnnealingSampler`（纯 NumPy Metropolis-Hastings Ising 退火） | 离线可跑，无需授权 |
| `sa` | `kaiwu.classical.SimulatedAnnealingOptimizer` | 已激活 Kaiwu SDK 授权 |
| `cim` | `kaiwu.cim.CIMOptimizer` + `PrecisionReducer`（兼容 `_kaiwu_compat`） | 已激活 Kaiwu SDK 授权与 CIM 真机机时 |
| `cd` | 纯经典 Gibbs 对比散度 | 离线可跑，无需授权 |

在已配置 Kaiwu SDK 授权（`kw.license.init(user_id, sdk_code)`）的环境中，可按如下命令切换：

```bash
# 使用 Kaiwu SDK 模拟退火采样器
python example/collaborative_filtering/train_cf.py --sampler sa

# 使用玻色量子相干光量子计算机（CIM）真机采样
python example/collaborative_filtering/train_cf.py \
  --sampler cim \
  --cim-task-name kpp_cf_rbm \
  --cim-task-mode sampling
```

关于 Kaiwu SDK 授权配置及 CIM 真机机时申请，请参阅仓库根目录 [README_ZH.md](../../README_ZH.md)。

## 实测定量基准对比

以下数据在 CPU 上使用默认合成潜在因子基准实测获得（`--seed 42`，400 名用户、80 件物品、6 个潜在偏好簇，共 5,483 条训练 / 788 条验证 / 1,568 条测试交互，观测密度 24.50%；严格按用户留出切分；Top-10 排序严格屏蔽所有训练与验证已见物品，相关性阈值 $\ge 4.0$）：

| 模型 | RMSE ↓ | MAE ↓ | P@10 ↑ | R@10 ↑ | NDCG@10 ↑ | CPU 耗时 |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 1.1888 | 0.9786 | 0.0349 | 0.2044 | 0.1095 | 0.00 s |
| TruncatedSVD (`r=6`) | 1.0474 | 0.8427 | 0.0564 | 0.3292 | 0.1844 | 0.23 s |
| RBM-CF (`cd`, `H=24`) | 0.8413 | 0.6755 | 0.0720 | 0.4318 | 0.2170 | 1.39 s |
| RBM-CF (`local_sa`, `H=24`) | **0.8404** | **0.6749** | 0.0689 | 0.4087 | 0.2061 | 3.63 s |

实测结论：
- 相比非个性化基线 `Popularity`，`RBM-CF (local_sa)` 测试集 RMSE 降低 **29.3%**、`NDCG@10` 提升 **+88.2%**；
- 相比矩阵分解基线 `TruncatedSVD (r=6)`，`RBM-CF (local_sa)` 测试集 RMSE 降低 **19.8%**、`NDCG@10` 提升 **+11.8%**；
- 训练过程中 `RBM-CF (local_sa)` 的训练 RMSE 从第 1 轮的 `1.2259` 平滑收敛至第 45 轮的 `0.9070`，并完整跑通 `RestrictedBoltzmannMachine.sample(sampler)` 的 Ising 矩阵转换与辅助自旋解码链路，可直接切换至 `sa` / `cim`。

## 运行单元测试

```bash
pytest tests/test_example_collaborative_filtering.py -v
```
