# 发行说明

本页记录 Kaiwu-PyTorch-Plugin 各版本的更新内容。

## 版本 0.3.0（2026-09-23）

### 新增

- GBRBM 高斯-伯努利受限玻尔兹曼机；
- MAIFS 模型无关特征选择（`FeatureSelectionWrapper`）；
- 新增 Docker 开发环境。
- 用量统计；

### 改造

- QVAE 重构为 `AutoEncoderBase` + config 驱动的工厂式构造；
- QDiffusion 进行了接口调整，移除 `EnergyBackboneAdapter` Protocol；
- dplm/qvae_mnist 示例由单体脚本拆分为 models/utils/workflows 分层结构。

### 文档

- 全量 rst 转 MyST Markdown，新增 10 篇理论基础、3 篇教程、贡献指南，数学渲染换 KaTeX。

## 版本 0.2.0（2026-07-07）

### 更新

- **新增 `QDiffusion`**
  在顶层插件接口中新增 `QDiffusion` 与 `QDiffusionConfig`，使用基于玻尔兹曼机作为能量模型优化扩散过程。

- **新增 `QDiffusion` 蛋白质生成示例**
  新增 `example/qdiffusion/` 示例目录，包括：
  - `simple/`：用于快速理解 `QDiffusion` API 的最小训练与生成示例
  - `dplm/`：基于 DPLM 的完整蛋白质序列生成工作流，涵盖模型构建、训练、引导生成、checkpoint 复现与评估

- **新增 Q-VAE 单细胞表征学习示例**
  新增 `example/qvae_cell/` 示例，提供单细胞表征学习完整流程，包括数据读取、Q-VAE 训练、潜在表示提取、可视化、能量分析以及下游 benchmark 评估。

- **新增 RISC-V 64 平台预编译 wheel**
  提供 kaiwu 1.3.1 / kpp 0.2.0 在 RISC-V 64 平台的预编译 wheel，安装方式见下文。

### RISC-V 64 安装

要求：RISC-V 64 + Python 3.10 + glibc ≥ 2.31。

```bash
pip install --upgrade pip
pip install requests pyjwt portion Deprecated nest-asyncio

pip install https://github.com/qboson/kaiwu-pytorch-plugin/releases/download/v0.2.0/cffi-2.1.1-cp310-cp310-linux_riscv64.whl
pip install https://github.com/qboson/kaiwu-pytorch-plugin/releases/download/v0.2.0/cryptography-50.0.0-cp310-abi3-linux_riscv64.whl
pip install https://github.com/qboson/kaiwu-pytorch-plugin/releases/download/v0.2.0/numpy-2.1.1-cp310-cp310-manylinux_2_31_riscv64.whl
pip install https://github.com/qboson/kaiwu-pytorch-plugin/releases/download/v0.2.0/pandas-2.2.2-cp310-cp310-manylinux_2_31_riscv64.whl
pip install https://github.com/qboson/kaiwu-pytorch-plugin/releases/download/v0.2.0/torch-2.7.0-cp310-cp310-manylinux_2_39_riscv64.whl
pip install --no-deps https://github.com/qboson/kaiwu-pytorch-plugin/releases/download/v0.2.0/kaiwu-1.3.1-cp310-cp310-manylinux_2_31_riscv64.manylinux_2_39_riscv64.whl
pip install --no-deps https://github.com/qboson/kaiwu-pytorch-plugin/releases/download/v0.2.0/kaiwu_torch_plugin-0.2.0-cp310-cp310-manylinux_2_31_riscv64.manylinux_2_39_riscv64.whl
```

验证：

```python
import torch; print('torch', torch.__version__)
import kaiwu; print('kaiwu', kaiwu.__version__)
from kaiwu.torch_plugin import QDiffusion, QVAE, RestrictedBoltzmannMachine
print('kpp OK')
```

> 注：kaiwu / kpp 因 PyPI 上 riscv64 依赖不全，需 `--no-deps`；numpy 须先于 pandas。

## 版本 0.1.1（2026-04-01）

### 主要新增功能

- 依赖许可证检查
- PyPI 发布流水线
- 文档与说明更新
- 调整BM接口和矩阵转换逻辑

## 版本 0.1.0（2025-12-24）

**新功能**

- 依赖许可证检查
- PyPI 发布流水线


**改进**

- 文档与说明更新
- 调整BM接口和矩阵转换逻辑


## 版本 0.0.2（2025-11-06）

**新功能**

- 新增文档框架，支持 Sphinx 构建
- 新增 DBN（深度信念网络）模块，支持多层 RBM 堆叠
- 新增 Q-VAE 示例，支持量子变分自编码器训练

**改进**

- 恢复并优化 RBM 模块实现
- 更新示例代码 README 文档
- 改进代码注释，提升可读性

**文档**

- 更新中英文 README 文档
- 添加更详细的安装说明
- 补充示例代码说明

**修复**

- 修复 DBN 模块中的已知问题
- 修复 pylint 检查发现的代码问题

## 版本 0.0.1（2025-10-15）

**初始版本**

Kaiwu-PyTorch-Plugin 的首个正式发布版本，提供以下核心功能：

**核心模块**

- `RestrictedBoltzmannMachine`：受限玻尔兹曼机实现
- `BoltzmannMachine`：全连接玻尔兹曼机实现
- `AbstractBoltzmannMachine`：抽象基类，支持自定义扩展

**主要特性**

- 支持 PyTorch 原生接口，可使用标准优化器（SGD、Adam 等）
- 支持 Kaiwu SDK 采样器，包括模拟退火和量子采样
- 支持 GPU 加速训练
- 提供完整的示例代码

**示例**

- `rbm_digits`：手写数字识别示例
- `qvae_mnist`：Q-VAE MNIST 生成示例
- `bm_generation`：玻尔兹曼机数据生成示例

**环境要求**

- Python 3.10
- PyTorch 2.7.0
- Kaiwu SDK v1.2.0+
