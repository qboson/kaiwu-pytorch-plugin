# 语言版本：[中文](README_ZH.md) | [English](README.md)

# 量子启发式金融投资组合优化（QUBO/Ising + RBM）

本示例基于 `kaiwu-pytorch-plugin` 实现了一套离线一键可复现的**基数约束 Markowitz 均值-方差投资组合优化**行业案例。项目将离散资产选择（$\sum_i x_i = K$）严谨构建为 QUBO 并通过 `kaiwu.torch_plugin._kaiwu_compat.qubo_matrix_to_ising_matrix` 映射至 Ising 哈密顿量，结合 `kaiwu.torch_plugin.RestrictedBoltzmannMachine` 建模资产联合状态情景以增强协方差估计，并在严格按时间滚动切分（零未来函数泄漏）的样本外测试期上，与 **Equal-Weight (`1/N`)**、**Greedy Top-K Sharpe** 及 **Exact Brute-Force 精确全局最优解**进行公平对比。

## 算法原理

给定严格在训练期估计的年化期望收益向量 $\mu \in \mathbb{R}^N$、年化协方差矩阵 $\Sigma \in \mathbb{R}^{N \times N}$、风险厌恶系数 $\gamma \ge 0$ 与目标持仓基数 $K \in \{1, \dots, N-1\}$（等权配置 $w_i = x_i / K$，$x \in \{0, 1\}^N$），目标函数为：

$$\min_{x \in \{0, 1\}^N} \;\; \frac{\gamma}{K^2} x^\top \Sigma x - \frac{1}{K} \mu^\top x + \lambda \left(\sum_{i=1}^N x_i - K\right)^2$$

利用二值变量恒等式 $x_i^2 = x_i$ 展开得到对称 QUBO 形式 $x^\top Q x + \lambda K^2$：

- 对角项：$Q_{ii} = \frac{\gamma}{K^2} \Sigma_{ii} - \frac{1}{K} \mu_i + \lambda (1 - 2K)$
- 非对角项：$Q_{ij} = \frac{\gamma}{K^2} \Sigma_{ij} + \lambda \quad (i \neq j)$

通过 `qubo_matrix_to_ising_matrix(Q)`（兼容 Kaiwu SDK 1.3.1 / 1.4.0 / 1.4.3）导出 $(N+1) \times (N+1)$ Ising 矩阵 $J$ 与常数偏置 $c$。对任意二值状态 $x \in \{0, 1\}^N$ 及两种辅助自旋极性 $s_{\text{aux}} \in \{-1, +1\}$（$s_i = s_{\text{aux}}(2x_i - 1)$），均严格满足能量等价关系：

$$x^\top Q x = -s^\top J s + c$$

此外，`RBMScenarioEnhancer` 仅在训练期二值化收益状态上调用 `RestrictedBoltzmannMachine`（`rbm.sample(sampler)` + `rbm.objective(s_pos, s_neg)`）学习资产联合共动分布，并将 RBM 生成情景的相关结构与经验协方差加权融合，用于协方差正则化与下行情景分析。

## 目录结构

```text
example/financial_portfolio/
├── __init__.py       # 包公共接口导出
├── dataset.py        # 多行业牛熊机制收益生成器、本地 CSV 加载器、严格时间顺序切分
├── qubo.py           # Markowitz QUBO/Ising 构造、辅助自旋解码与精确能量等价核验
├── optimizer.py      # LocalIsingPortfolioSampler、Kaiwu SA/CIM 工厂、RBMScenarioEnhancer
├── baselines.py      # Equal-Weight (1/N)、Greedy Top-K Sharpe、精确枚举最优解求解器
├── metrics.py        # 样本外年化收益、波动率、夏普比率、最大回撤、可行性与最优性间隙
├── benchmark.py      # 基准调度、CLI 参数解析与结果对比表格式化
├── run_portfolio.py  # 可直接运行的命令行入口
├── README.md         # 英文说明文档
└── README_ZH.md      # 中文说明文档
```

## 依赖环境

本示例**仅依赖 `kaiwu-torch-plugin` 核心依赖**（`torch`、`numpy`、`kaiwu`），默认模式无需任何外部数据下载即可一键运行：

```bash
pip install -r requirements/requirements.txt
pip install .
```

## 快速开始

运行默认离线 16 资产、$K=5$ 基准（CPU 约 2 秒完成）：

```bash
python example/financial_portfolio/run_portfolio.py
```

快速冒烟运行（`--fast`）并导出结构化 JSON 结果：

```bash
python example/financial_portfolio/run_portfolio.py --fast --output-json /tmp/portfolio_results.json
```

加载本地 CSV 资产收益文件运行（若为价格序列请附加 `--is-prices`）：

```bash
python example/financial_portfolio/run_portfolio.py \
  --dataset csv \
  --data-path /path/to/returns.csv \
  --target-k 5 \
  --train-ratio 0.6
```

## 切换至 Kaiwu SDK 模拟退火或相干光量子真机（CIM）

通过 `--sampler` 参数可在不修改模型代码的情况下统一切换 QUBO 求解与 RBM 采样的 Ising 后端：

| `--sampler` | 采样后端 | 运行前提 |
|---|---|---|
| `local_sa`（默认） | `LocalIsingPortfolioSampler`（Metropolis-Hastings + Kawasaki 2-spin 精炼） | 离线可跑，无需授权 |
| `sa` | `kaiwu.classical.SimulatedAnnealingOptimizer` | 已激活 Kaiwu SDK 授权 |
| `cim` | `kaiwu.cim.CIMOptimizer` + `PrecisionReducer`（兼容 `_kaiwu_compat`） | 已激活 Kaiwu SDK 授权与 CIM 真机机时 |

在已配置 Kaiwu SDK 授权（`kw.license.init(user_id, sdk_code)`）的环境中，可按如下命令切换：

```bash
# 使用 Kaiwu SDK 模拟退火求解器
python example/financial_portfolio/run_portfolio.py --sampler sa

# 使用玻色量子相干光量子计算机（CIM）真机求解
python example/financial_portfolio/run_portfolio.py \
  --sampler cim \
  --cim-task-name kpp_portfolio_qubo \
  --cim-task-mode optimization
```

关于 Kaiwu SDK 授权配置与 CIM 真机机时申请，请参阅仓库根目录 [README_ZH.md](../../README_ZH.md)。

## 实测定量基准对比

以下数据在 CPU 上使用默认确定性多行业合成市场数据实测获得（`--seed 42`，4 行业共 $N=16$ 资产、基数 $K=5$，严格时间切分 302 训练日 / 202 未来样本外测试日，$\gamma=2.5$，自动校准惩罚系数 $\lambda=0.3226$）：

- **QUBO-Ising 能量等价性最大绝对误差**：`9.770e-15`（覆盖 $s_{\text{aux}} \in \{-1, +1\}$ 双极性）。

| 策略 | 年化收益 ↑ | 年化波动 ↓ | 夏普比率 ↑ | 最大回撤 ↓ | 满足基数 | 最优性间隙 ↓ | CPU 耗时 |
|---|---:|---:|---:|---:|:---:|---:|---:|
| Equal-Weight (`1/N`，16 资产) | 12.48% | 13.89% | 0.916 | 7.29% | YES | N/A | 0.000 s |
| Greedy Top-K Sharpe (`K=5`) | 4.65% | 23.09% | 0.312 | 17.96% | YES | 96.54% | 0.000 s |
| Exact Brute-Force (`K=5`，4,368 组合) | 11.52% | **13.77%** | 0.861 | 9.39% | YES | **0.00%** | 0.007 s |
| QUBO-Ising (`local_sa`, `K=5`) | 11.52% | **13.77%** | 0.861 | 9.39% | YES | **0.00%** | 0.020 s |
| QUBO-Ising + RBM (`local_sa`, `K=5`) | 11.52% | **13.77%** | 0.861 | 9.39% | YES | **0.00%** | 1.875 s |

实测结论：
- `QUBO-Ising (local_sa)` 在 `0.020 s` 内以 **0.00% 最优性间隙**精确找到全部 4,368 个可行组合中的理论全局最优解，并 100% 满足 $K=5$ 基数约束；
- 相比容易陷入高相关单一行业集中风险的 `Greedy Top-K Sharpe (K=5)`，`QUBO-Ising` 将样本外年化波动率从 `23.09%` 降至 `13.77%`（**-40.4%**），最大回撤从 `17.96%` 降至 `9.39%`（**-47.7%**），夏普比率从 `0.312` 提升至 `0.861`（**+176.0%**）；
- `QUBO-Ising + RBM` 严格在训练期状态上调用 `RestrictedBoltzmannMachine` 完成协方差情景增强，同样稳健收敛至最优分散化 5 资产组合。

## 运行单元测试

```bash
pytest tests/test_example_financial_portfolio.py -v
```
