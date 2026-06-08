# KPP QVAE 单细胞表征学习

本目录提供一个基于 `kaiwu-pytorch-plugin` 的 QVAE 单细胞表征学习实现。流程包括读取单细胞表达矩阵、训练 QVAE、提取低维表征、计算 UMAP、分析能量分布，并用细胞类型标签评估聚类质量。

## 数据
示例使用数据可以在https://www.kaggle.com/datasets/redwoodz/immune-data下载

## 依赖
```
kaiwu==1.3.1
kaiwu-torch-plugin==0.1.0
torch==2.7.0
anndata
scanpy
leidenalg
scib_metrics
scgraph
```

## 文件结构

```text
kpp_qvae/
├── models.py                  # QVAEEncoder / QVAEDecoder / CellQVAE
├── trainer.py                 # 数据准备、模型构建、训练循环、表征和 energy 提取
├── visualization.py           # 训练曲线、UMAP 和 energy 图
├── train_qvae_cell.py         # 命令行入口
├── evaluate_clustering.py      # 旧版 Leiden 聚类评估入口
├── evaluate_benchmark.py       # 统一评估入口
├── scripts/
│   ├── train.sh               # 当前训练命令
│   ├── eval_clustering.sh                # 聚类评估命令
│   └── eval.sh           # 综合评估命令
└── README.md
```

## 训练

默认数据为：

```text
../immune_processed.h5ad
```

默认观测字段：

```text
batch_key = batch
labels_key = final_annotation
```

运行训练：

```bash
bash scripts/train.sh
```

快速检查：

```bash
python train_qvae_cell.py \
  --epochs 1 \
  --sampler-type sa \
  --loss-type mse \
  --representation q \
  --output-dir ./outputs_smoke
```

指定其他数据：

```bash
python train_qvae_cell.py \
  --data-path /path/to/data.h5ad \
  --epochs 100 \
  --output-dir ./outputs_custom
```

## 输出

训练脚本会在 `--output-dir` 下生成：

```text
immune_kpp_qvae_best.pth          # 最优模型权重
model_epoch*.pth                  # 按 --checkpoint-every 保存的中间权重
X_qvae.npy                        # QVAE 表征矩阵
immune_kpp_qvae.h5ad              # 写入 X_qvae 和 QVAE_Energy 的 AnnData
training_history.csv              # 训练指标
training_curves.png               # loss / reconstruction / KL 曲线
qvae_umap_labels_batches.png      # cell type 和 batch UMAP
qvae_energy_umap.png              # energy UMAP
qvae_energy_by_celltype.png       # cell type energy 分布
```

## 评估

统一评估入口为 `evaluate_benchmark.py`，通过 `--metrics` 选择要运行的评估项：

|评估方法|作用|
|----|----|
|clustering   |     Leiden 聚类指标：ARI, AMI, NMI, Homogeneity, FMI（越高越好）|
|classification |   Logistic Regression 下游细胞类型分类 |
|scib     |         scIB 生物保留和批次校正指标，即是否既保留了细胞类型结构，又消除了不必要的批次效应 |
|scgraph |          scGraph 图结构评价 |
|dpt  |             DPT 伪时序和轨迹一致性，即是否能产生合理的细胞发育/状态变化顺序 |
|all |              运行以上所有评估项 |

只跑聚类评估：

```bash
python evaluate_benchmark.py \
  --h5ad ./outputs/immune_kpp_qvae.h5ad \
  --rep-key X_qvae \
  --label-key final_annotation \
  --metrics clustering
```

跑分类评估：

```bash
python evaluate_benchmark.py \
  --h5ad ./outputs/immune_kpp_qvae.h5ad \
  --rep-key X_qvae \
  --label-key final_annotation \
  --metrics classification
```

跑多个评估项：

```bash
python evaluate_benchmark.py \
  --h5ad ./outputs/immune_kpp_qvae.h5ad \
  --rep-key X_qvae \
  --label-key final_annotation \
  --batch-key batch \
  --metrics clustering,classification,scib,scgraph
```

跑 DPT：

```bash
python evaluate_benchmark.py \
  --h5ad ./outputs/immune_kpp_qvae.h5ad \
  --rep-key X_qvae \
  --label-key final_annotation \
  --metrics dpt \
  --root-cell-type HSPCs
```

`bash scripts/eval.sh` 等价于运行 `--metrics clustering`。结果会保存为对应的 CSV，并汇总到：

```text
<output_dir>/X_qvae_benchmark_summary.csv
```
## 关键参数

```text
--latent-dim      默认 256
--num-visible     默认 128
--num-hidden      默认 128
--sampler-type    sa 或 cim
--loss-type       默认 mse；可选 mse 或 bernoulli
--representation  默认 q；可选 zeta 或 q
--kl-beta         默认 1e-5
--dist-beta       默认 10.0
--lr              默认 1e-4
--rbm-lr          默认 1e-3
```
