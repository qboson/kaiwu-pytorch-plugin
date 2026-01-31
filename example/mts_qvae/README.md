# MTS（线粒体靶向序列）论文复现：VAE vs QBM-VAE(QVAE)

本目录用于完成 kaiwu-pytorch-plugin Issue #26：
- 复现论文 *Design of diverse, functional mitochondrial targeting sequences across eukaryotic organisms using variational autoencoder* 的 **MTS-VAE** 训练/生成流程
- 在相同数据集与相同输入表示下，将论文中的 **VAE 先验（标准正态）** 替换为本仓库实现的 **QVAE（以 Boltzmann Machine/QBM 作为潜变量先验）**
- 输出 **可重复运行** 的训练日志与基础对比指标（不做学术造假：不会宣称复现了论文中依赖外部闭源/在线服务的所有结果）

## ⚠️ 重要限制说明（使用前必读）

> **RandomIsingSampler 是占位采样器**
>
> 本示例默认使用的 `RandomIsingSampler` **仅返回随机二值解**，并不会真正求解 Ising 优化问题。
> 它的存在仅为了让示例代码在没有安装 Kaiwu 真实求解器的情况下也能运行通过。
>
> ⚠️ **如果您需要科学上有效的 QVAE vs VAE 性能对比，必须使用真实的 Ising 求解器！**

### 如何使用真实的 Ising 求解器

1. **安装 Kaiwu SDK**：请参考 [玻色量子开发者平台](https://github.com/qboson) 获取完整的 Kaiwu SDK
2. **替换采样器**：在 `train_qvae.py` 中将 `RandomIsingSampler` 替换为 Kaiwu 提供的真实采样器，例如：
   ```python
   # 使用 Kaiwu SDK 的模拟退火采样器
   from kaiwu import SimulatedAnnealingSampler
   sampler = SimulatedAnnealingSampler(...)
   
   # 或使用量子退火采样器（需连接量子硬件）
   from kaiwu import QuantumAnnealingSampler
   sampler = QuantumAnnealingSampler(...)
   ```
3. **重新训练**：使用真实采样器重新训练 QVAE 模型以获得有效的对比结果

## 数据来源（论文官方）
- 代码：https://github.com/Zhao-Group/MTS-VAE
- 数据集（Zenodo）：https://zenodo.org/records/14590156
  - 文件：`Datasets.zip`
  - md5：`e24a1dd48aef7bd5bb416d1e1fa9d257`

论文中描述的训练数据规模/预处理要点：
- MTS 数据集规模：56,660 条
- 长度过滤：11–69 aa
- 在 C 端追加 `$` 作为 cleavage 标记
- 统一 pad 到长度 70，pad 字符为 `0`
- one-hot 字母表顺序（严格按原仓库脚本）：`FIWLVMYCATHGSQRKNEPD$0`（共 22 个 token）
- VAE baseline 结构：FC(1540→512) + latent=32 + FC(32→512→1540)

## 快速开始

### 0) 一键全自动（推荐）

```bash
python example/mts_qvae/run_all.py
```

常用参数：
- 只做快速冒烟测试（更快验证流程是否跑通）：

```bash
python example/mts_qvae/run_all.py --vae-epochs 1 --qvae-epochs 1 --n-samples 200
```

- 完整复现默认训练轮数（与脚本默认一致）：

```bash
python example/mts_qvae/run_all.py --vae-epochs 35 --qvae-epochs 35 --n-samples 1000
```

### 1) 下载并解压数据

在仓库根目录（或任意位置）执行：

```bash
python example/mts_qvae/download_datasets.py --out-dir example/mts_qvae/data/zenodo
```

下载完成后会得到：
- `example/mts_qvae/data/zenodo/Datasets.zip`
- 解压后的 `MTS/` 与 `Dual/` 目录（保持原始结构）

### 2) 训练 VAE baseline（论文同结构）

```bash
python example/mts_qvae/train_vae_baseline.py --data-root example/mts_qvae/data/zenodo
```

### 3) 训练 QVAE（QBM-VAE 替换先验）

```bash
python example/mts_qvae/train_qvae.py --data-root example/mts_qvae/data/zenodo
```

### 4) 生成与评估（基础指标）

```bash
python example/mts_qvae/evaluate_compare.py --data-root example/mts_qvae/data/zenodo \
  --vae-ckpt example/mts_qvae/outputs/vae_baseline.pt \
  --qvae-ckpt example/mts_qvae/outputs/qvae.pt

额外可复现输出（不依赖外部工具）：

```bash
python example/mts_qvae/evaluate_compare.py --data-root example/mts_qvae/data/zenodo \
  --vae-ckpt example/mts_qvae/outputs/vae_baseline.pt \
  --qvae-ckpt example/mts_qvae/outputs/qvae.pt \
  --save-fasta --plots
```

会在 `example/mts_qvae/outputs/` 下额外写出：
- `vae_samples.fasta` / `qvae_samples.fasta`
- `figs/length_hist.png`、`figs/aa_composition.png`
```

## 关于“对比验证”的说明（避免学术造假）

论文中的功能性评估包含 DeepLoc 2.0 等外部工具/服务，以及一些额外依赖（如 TargetP 2.0 本地软件包、UniRep 等）。
本目录默认实现的是：
- 严格按论文/原仓库脚本的输入表示与 baseline VAE 结构训练
- 以同样输入表示训练 QVAE（仅替换潜变量先验）
- 输出可核对的基础指标：训练/验证损失、生成序列的有效率/去重率、与训练集完全相同序列的比例、长度与 token 频率统计等

如果你本地具备 DeepLoc 2.0/TargetP 2.0 的可运行环境，可以在此基础上扩展 `evaluate_compare.py` 以复现论文的更完整评估。

注意：本仓库/本示例不会静默下载或调用在线服务来“补齐”论文评估结果；未检测到外部工具时会明确提示缺失。

### DeepLoc2 / TargetP2（本地版）对齐的现实限制（重要）

- DeepLoc 2.0 与 TargetP 2.0 的“本地/离线软件包”来自 DTU Health Tech 的学术下载申请渠道，需要同意许可条款并通过机构邮箱获取下载链接。
- TargetP 2.0 官方下载页面标注的平台为 Linux；在 Windows 上通常需要 WSL2 或 Linux 机器/容器环境。
- 因为公开网页不提供可核验的 CLI 参数/输出列定义，本仓库不会“猜测”它们。

本示例提供的是“外部工具调用钩子”：你拿到官方软件包后，按官方说明确定命令行，然后通过 `--targetp-cmd/--deeploc-cmd` 显式传入。

#### 外部工具调用钩子用法（不提供默认命令，避免编造）

支持占位符：`{in_fasta}`、`{out_dir}`、`{out_path}`。

输出解析与“论文式统计图”也已接好，但需要你提供“真实输出格式”的最小信息：
- 如果外部工具能输出 TSV/CSV（short output），请准备一份 schema：参考 `external_schema.template.json`，把 `category_col` 改成你输出里表示“预测类别”的那一列名。
- 然后在评估时传入：`--external-schema example/mts_qvae/external_schema.template.json` 以及 `--targetp-out/--deeploc-out` 指向实际输出文件。

为什么需要 schema？因为 DTU 的离线包属于许可分发，公开网页不提供可核验的列名/输出定义，本仓库不会凭空假设列名。

示例（命令仅演示占位符传参方式，不代表真实 TargetP/DeepLoc CLI）：

```bash
python example/mts_qvae/evaluate_compare.py --data-root example/mts_qvae/data/zenodo \
  --vae-ckpt example/mts_qvae/outputs/vae_baseline.pt \
  --qvae-ckpt example/mts_qvae/outputs/qvae.pt \
  --save-fasta --plots \
  --targetp-cmd <your_targetp_tokens_here> \
  --deeploc-cmd <your_deeploc_tokens_here>
```

拿到 DTU 离线包后，你可以把“官方 README/--help 输出”贴给我，我会基于真实文档把命令与输出解析（TSV/JSON）做成自动统计与图表。

### 我为什么不能“自己操作把 DTU 工具装好并写死解析”

- DeepLoc 2.0 / TargetP 2.0 的离线包需要你以机构邮箱提交学术下载申请并同意许可条款后获得下载链接；我无法替你完成这一授权步骤。
- TargetP 2.0 官方标注为 Linux 平台；在 Windows 上通常需要 WSL2/Linux。
- 在没有离线包文档/输出样例前，写死 CLI flags 或输出列名会变成“猜测”，这违反你的“不造假”要求。
