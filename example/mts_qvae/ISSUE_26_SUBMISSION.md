# Issue #26 提交说明（不造假版本）

本仓库已新增 [example/mts_qvae](example/mts_qvae) 用于复现论文的 MTS 生成流程，并用本仓库 QVAE（QBM 先验）替换 baseline VAE 先验，提供可重复运行的训练与对比评估脚本。

## 你需要交付什么？

对 GitHub Issue #26 来说，**最重要的交付物是代码与可复现流程**（脚本+文档），而不是把训练好的模型权重或 547MB 数据集提交进仓库。

提交内容建议包含：
- 代码：`example/mts_qvae/*`（下载/训练/评估/外部评估钩子/解析框架）
- 文档：`example/mts_qvae/README.md`、本文件
- 兼容入口：`train_qvae.py`（满足 Issue 中常见的运行方式）
- `.gitignore`：确保不提交 `example/mts_qvae/data/` 和 `example/mts_qvae/outputs/`

不建议提交：
- `example/mts_qvae/data/`（Zenodo 数据集解压内容）
- `example/mts_qvae/outputs/`（ckpt、生成序列、图表、metrics）

## 你完成任务了吗？

- **代码复现与对比框架：已完成**（下载→训练 VAE→训练 QVAE→对比评估，一键 `run_all.py` 也可跑通）。
- **论文中的 DeepLoc2/TargetP2 功能评估：已提供“可选集成框架”，但无法在未获得 DTU 离线包/文档情况下写死 CLI/列名**（否则属于猜测，有造假风险）。

因此：
- “完成 Issue 的可复现代码交付” ✅
- “完全复刻论文所有外部工具评估并给出一致结论” 取决于你是否拿到离线包并能运行（我们已留好全自动接入入口）

## 最小复现命令（供 Issue 评论区粘贴）

在仓库根目录运行：

1) 下载并解压（会校验 md5）：

```bash
python example/mts_qvae/download_datasets.py --out-dir example/mts_qvae/data/zenodo
```

2) 训练 VAE baseline：

```bash
python example/mts_qvae/train_vae_baseline.py --data-root example/mts_qvae/data/zenodo
```

3) 训练 QVAE：

```bash
python train_qvae.py --data-root example/mts_qvae/data/zenodo
```

4) 对比评估 + FASTA + 图表：

```bash
python example/mts_qvae/evaluate_compare.py --data-root example/mts_qvae/data/zenodo \
  --vae-ckpt example/mts_qvae/outputs/vae_baseline.pt \
  --qvae-ckpt example/mts_qvae/outputs/qvae.pt \
  --save-fasta --plots
```

一键全自动：

```bash
python example/mts_qvae/run_all.py
```

冒烟测试：

```bash
python example/mts_qvae/run_all.py --vae-epochs 1 --qvae-epochs 1 --n-samples 200
```

## 如何提交（推荐 PR 流程）

```bash
# 1) 进入仓库根目录
cd kaiwu-pytorch-plugin-main/kaiwu-pytorch-plugin-main

# 2) 新建分支
git checkout -b issue-26-mts-qvae

# 3) 确认不会把数据/输出提交进去
git status

# 4) 跑单测（可选但推荐）
pytest -q

# 5) 添加变更
git add -A

# 6) 提交
git commit -m "Add MTS QVAE reproduction example (Issue #26)"

# 7) 推送到你的 fork
git push -u origin issue-26-mts-qvae
```

然后去 GitHub 打开 PR，标题建议：
- `Issue #26: Add MTS VAE vs QVAE reproduction pipeline`

PR/Issue 描述建议包含：
- 数据来源（Zenodo 记录号与 md5）
- 运行命令（上面“最小复现命令”）
- 说明外部工具（DeepLoc2/TargetP2）的许可限制与本仓库的“钩子式集成”策略
