# 语言版本：[中文](README_ZH.md) | [English](README.md)

# `example/dbm`

本目录提供 `DeepBoltzmannMachine`（深度玻尔兹曼机，DBM）的离线可运行示例，演示多模态二值模式学习、去噪重建以及基于联合 Ising 矩阵的多层一次性采样流程。

## 模型简介

与逐层贪婪堆叠的深度信念网络（`UnsupervisedDBN`）不同，`DeepBoltzmannMachine` 在可见层 `v` 与多层隐层 `h^(1), ..., h^(L)` 之间建立统一的无向双向耦合哈密顿量：

$$E(v, h^{(1)}, \dots, h^{(L)}) = - b_v^T v - \sum_{\ell=1}^{L} b_{h^{(\ell)}}^T h^{(\ell)} - v^T W^{(1)} h^{(1)} - \sum_{\ell=2}^{L} (h^{(\ell-1)})^T W^{(\ell)} h^{(\ell)}$$

- **正相推断**：使用变分均值场推断 `mean_field_inference`，每步层坐标更新严格保证变分自由能单调不增。
- **负相联合采样**：`to_ising_matrix()` / `get_ising_matrix()` 将可见层与所有隐层统一映射为单个 `(num_nodes + 1, num_nodes + 1)` 联合 Ising 矩阵，使 Kaiwu 模拟退火或光量子 CIM 可一次性对 `(v, h^(1), ..., h^(L))` 执行多体联合采样。

## 快速开始

本示例完全离线运行，无需下载额外数据集或安装额外依赖。在仓库根目录执行：

```bash
pip install -e .
python example/dbm/run_dbm.py
```

可选参数：

- `--epochs`：训练轮数（默认 `80`）
- `--lr`：Adam 学习率（默认 `0.1`）
- `--seed`：随机种子（默认 `42`）
- `--sampler`：Ising 采样器后端（默认 `local` 免授权 NumPy 模拟退火；可选 `sa` 使用 Kaiwu `SimulatedAnnealingOptimizer`，或 `cim` 使用 `CIMOptimizer` + `PrecisionReducer`，需通过 `USER_ID`/`SDK_CODE` 初始化 Kaiwu 授权）

## 示例说明

`run_dbm.py` 在 4 类 12 位结构化多模态二值模式（两两汉明距离为 6）上训练双隐层 DBM（`num_visible=12`, `hidden_dims=(8, 4)`），在 CPU 上数秒即可完成并输出训练前后对比指标：

- **变分自由能间隔** `F_MF(v_ood) - F_MF(v_in)`
- 面对 2 比特翻转受损输入时的**多层均值场去噪重建准确率**（逐位与精确匹配）
- 基于 `dbm.sample(sampler)` 的**联合 Ising 采样模式命中率**

CPU 实测输出（`--seed 42`, `--sampler local`，约 4.1 秒）：

| 阶段 | `vfe_gap` | `denoise_bit_acc` | `denoise_exact_acc` | `gen_valid_rate` |
| --- | ---: | ---: | ---: | ---: |
| 初始（Init） | `0.000` | `47.92%` | `0.00%` | `0.00%` |
| 训练后（80 轮） | `13.200` | `100.00%` | `100.00%` | `100.00%` |
