# 语言版本：[中文](README_ZH.md) | [English](README.md)

# `example/qdiffusion`

本目录提供公开 `Q-Diffusion` 模块的示例与工作流脚本，重点展示如何将通用
`Q-Diffusion` 核心接入 DPLM 骨干网络，用于蛋白质离散扩散生成任务。

## 数据

[example/qdiffusion 使用的数据](https://www.uniprot.org/proteomes/UP000005640)

下载数据后，请将 FASTA 文件放到：

```text
example/qdiffusion/data/UP000005640_9606.fasta
```

如果你使用自己的数据，也可以放在其他位置，但需要在对应脚本中将
`fasta_path` 或 `reference_fasta` 改成你的实际路径。

## 快速开始

```bash
pip install -r example/qdiffusion/requirements.txt
python example/qdiffusion/simple/simple_train_example.py
python example/qdiffusion/simple/simple_generate_example.py
```

以上命令适用于两种常见使用方式：

- 开发态源码运行：直接在仓库根目录执行
- 安装态运行：先执行 `pip install -e .`，再运行相同命令

## 目录结构与整体关系

这个示例目录围绕一个清晰的边界组织：

- `src/kaiwu/torch_plugin/qdiffusion.py`：通用的 `Q-Diffusion` 核心实现
- `example/qdiffusion/dplm/`：将 DPLM checkpoint 适配到通用 `Q-Diffusion`
- `example/qdiffusion/simple/`：最小可运行示例，便于先理解 API

主装配路径如下：

1. 脚本调用 `build_qdiffusion(...)`
2. `dplm/utils/dplm_builder.py` 加载 DPLM proposal backbone 和 energy backbone
3. `dplm/models/` 基于特征编码器构建条件能量打分模块
4. builder 组装出一个通用的 `Q-Diffusion(...)`
5. 脚本通过 `objective(...)` 执行训练，或通过 `generate(...)` 执行推理

## Simple

`simple/` 是最小可运行入口：

- `simple/simple_train_example.py`：最小训练目标与优化循环
- `simple/simple_generate_example.py`：最小迭代生成流程

这两个脚本底层仍然会使用 DPLM，但只通过 `dplm/` 里的 example-side
factory helper 间接接入。

如果你想先理解公开 API，而不想一开始就进入完整实验脚本，建议先看
`simple/`。

## dplm

`dplm/` 目录包含蛋白案例相关的适配层和完整工作流：

- `dplm/utils/dplm_builder.py`：蛋白案例到 `Q-Diffusion` 的组装工具
- `dplm/models/`：模型侧代码，按 backbone、energy reranker 和私有 ESM patch 分层
- `dplm/utils/`：工作流侧工具，负责 FASTA I/O、checkpoint 和评估指标
- `dplm/workflows/`：真正的训练和 ESM2 评估实现
- `dplm/train_workflow.py`：完整训练与评估工作流的兼容入口
- `dplm/eval_esm2_distances.py`：ESM2 distance 评估的兼容入口

如果你想看真正的实现链路，建议从 `dplm/workflows/train.py` 开始读。

它的端到端流程如下：

1. 读取并过滤 FASTA 记录
2. 将数据划分为 train / validation / test
3. 构建一个 `Q-Diffusion` 生成器，proposal 侧使用 DPLM，energy 侧使用条件能量重排模块
4. 将蛋白质序列 tokenize 为 `targets`
5. 在 epoch 循环中调用 `generator.objective({"targets": ...})`
6. 优化 `energy_objective.mean()`，主要训练 energy reranking 分支
7. 保存轻量 checkpoint，包括 `energy_encoder`、`feature_projector`、energy backend 权重、`energy_head` 和 `vocab_proj`
8. 重建 baseline 与 guided 两套生成器用于测试时生成
9. 对 baseline 和 guided 的输出进行对比并生成报告

## 示例 ESM2 距离结果

这个 DPLM 引导工作流提供 `dplm/eval_esm2_distances.py`，用于在 embedding
层面比较生成序列与 reference proteome 的距离。当前一组示例评估使用了：

- reference: `data/UP000005640_9606.fasta`
- esm2 model: `esm2_t33_650M_UR50D`
- pair mode: `order`
- pooling: `mean`

| label | pairs | mean cosine dist | median cosine dist | mean l2 dist | median l2 dist |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 20 | 0.232665 | 0.192503 | 5.199004 | 4.926634 |
| MLP | 200 | 0.227007 | 0.205144 | 4.945239 | 4.803553 |
| guided | 20 | 0.187927 | 0.159195 | 4.679255 | 4.432099 |

在这次评估中，guided 相比 baseline 的改善为：

- mean cosine distance：`-0.044738`
- mean L2 distance：`-0.519750`

由于这两项指标越低代表生成结果在 ESM2 embedding space 中越接近参考集，
因此这组结果说明能量引导路径在该次评估样本上带来了可观的质量提升。

## `Q-Diffusion.objective(...)` 的训练路径

在 `Q-Diffusion.objective(...)` 内部，主要训练链路如下：

1. 从干净的 `targets` 开始
2. 将其扰动为带噪声的 `x_t`
3. 通过 proposal model 生成 logits
4. 从 logits 中采样负样本候选
5. 用条件能量重排模块对正样本与负样本进行打分
6. 返回 `energy_objective` 及相关中间张量给外层训练循环

## 共享资源

- `data/UP000005640_9606.fasta`：示例使用的 FASTA 数据
- `graph/*`：从原始案例笔记中整理出的图示资源

## 说明

- 本目录是 example-only；可复用的库代码位于 `src/kaiwu/torch_plugin/qdiffusion.py`
- 用户应从 `kaiwu.torch_plugin` 导入通用 `Q-Diffusion` 核心
- DPLM 加载逻辑不属于正式 `src` 公共 API，而是此目录中的 example-side compatibility layer
- 当前示例中的 guided 路径本质上是 `DPLM proposal + BM energy reranker`
- `simple/` 与 `dplm/` 建议配合阅读：`simple/` 展示 API 表层，`dplm/` 展示完整实验工作流
