# Jet Clustering Paper Reproduction with Kaiwu SDK

使用 Kaiwu SDK 复现论文 "A novel quantum realization of jet clustering in high-energy physics experiments"

## 项目介绍

本项目是玻色量子「开源贡献季」#74 任务的完整实现，使用 QUBO (Quadratic Unconstrained Binary Optimization) 方法将高能物理中的喷注聚类问题转化为可在量子计算机上求解的优化问题。

### 论文信息
- **标题**: A novel quantum realization of jet clustering in high-energy physics experiments
- **作者**: Qing-Yu Men, Teng-Fei Li, Ping Ao, Xin Shi, Liang-Liang Wang
- **核心贡献**: 首次将喷注聚类问题转化为 QUBO 问题并在量子计算机上实现

## 功能特点

### 算法实现
- **QAOA 量子优化**: 完整的量子近似优化算法实现，支持多层深度 (p=1,3,5)
- **kt 算法族**: anti-kt、Durham/e+e- kt、Cambridge/Aachen 算法
- **k-Means 聚类**: 适用于喷注物理的自适应 k-Means 实现
- **QUBO 建模**: 完整实现论文公式 7-10

### 论文复现
完整复现 Figure 2 的全部子图：
- **(a)** QAOA 深度对比 (k=6, depth=1,3,5)
- **(b)** k 值对比 (depth=3, k=2,4,6,7,8)
- **(c)** 算法对比 (QAOA vs e+e- kt vs k-Means)
- **(d)** 量子电路可视化
- **(e)** Quafu 量子硬件对比

### Web 可视化系统
交互式 Web 仪表板，支持：
- 参数实时调整
- 算法对比分析
- η-φ 平面可视化
- 性能指标展示

## 项目结构

```
QUBO问题/
├── src/                          # 核心源代码
│   ├── algorithms/               # 算法实现
│   │   ├── qaoa.py              # QAOA 量子优化算法
│   │   ├── kt_algorithm.py      # kt 算法族
│   │   └── kmeans_jet.py        # 喷注 k-Means
│   ├── physics/                  # 物理计算
│   │   ├── jet_physics.py       # 喷注物理
│   │   └── metrics.py           # 性能指标
│   ├── simulation/               # 事件模拟
│   │   └── event_generator.py   # 事件生成器
│   └── visualization/            # 可视化
│       ├── paper_figures.py     # 论文图表复现
│       └── circuit_visualization.py  # 量子电路图
├── experiments/                  # 实验运行
│   ├── run_experiments.py       # 主实验脚本
│   └── results/                 # 结果输出
├── web/                          # Web 仪表板
│   ├── app.py                   # Flask 后端
│   ├── templates/               # HTML 模板
│   └── static/                  # CSS/JS 资源
├── kaiwu/                        # Kaiwu SDK 模拟
├── jet_clustering_qubo.py       # 原始实现
├── requirements.txt             # 依赖列表
└── README.md                    # 本文件
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行完整实验

```bash
python experiments/run_experiments.py --figure all
```

这将执行所有 Figure 2 实验并在 `experiments/results/` 目录生成图表。

### 运行单个图表实验

```bash
# Figure 2(a) - QAOA 深度对比
python experiments/run_experiments.py --figure 2a

# Figure 2(c) - 算法对比
python experiments/run_experiments.py --figure 2c
```

### 启动 Web 仪表板

```bash
cd web
python app.py
```

打开浏览器访问 http://localhost:5000

### 运行原始演示

```bash
python jet_clustering_qubo.py
```

## 核心算法说明

### QUBO 建模 (论文公式 7-10)

**决策变量**:
```
x_{ij} ∈ {0, 1}  —— 粒子 i 是否属于喷注 j
```

**目标函数** (最小化喷注内距离):
```
H_obj = Σ_k Σ_{i<j} d_{ij} × x_{ik} × x_{jk}
```

**约束条件** (每个粒子只属于一个喷注):
```
Σ_j x_{ij} = 1, ∀i
```

**距离度量** (kt 算法):
```
d_{ij} = min(k_{Ti}^{2p}, k_{Tj}^{2p}) × ΔR_{ij}² / R²
```

### 性能指标

主要指标：**平均角度 (rad)** —— 重建喷注与真实夸克方向的夹角
- 越小表示聚类效果越好
- 论文 Figure 2 所有子图的 y 轴指标

## 与论文对比

| 方面 | 论文方法 | 本实现 |
|------|----------|--------|
| 量子硬件 | D-Wave Advantage | Kaiwu CIM / 模拟 |
| 变量类型 | 二进制 QUBO | 二进制 QUBO |
| 距离度量 | kt 算法 (p=-1,0,1) | 完整支持 |
| 约束处理 | 惩罚函数法 | PenaltyMethodSolver |
| 经典对比 | 模拟退火 | SimulatedAnnealingOptimizer |

## 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 测试单个模块
python src/algorithms/qaoa.py
python src/algorithms/kt_algorithm.py
python src/algorithms/kmeans_jet.py
```

## 参考资料

- **论文**: A novel quantum realization of jet clustering in high-energy physics experiments
- **Kaiwu SDK**: https://kaiwu-sdk-docs.qboson.com/
- **玻色量子**: https://www.qboson.com/
- **GitHub**: https://github.com/qboson/kaiwu-pytorch-plugin
