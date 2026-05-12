# KPP QVAE 单细胞表征学习

本目录提供一个基于 `kaiwu-pytorch-plugin` 的 QVAE 单细胞表征学习实现。流程包括读取单细胞表达矩阵、训练 QVAE、提取低维表征、计算 UMAP、分析能量分布，并用细胞类型标签评估聚类质量。
## 依赖
```
kaiwu==1.1.0a0
kaiwu-torch-plugin==0.1.0
torch==2.7.0
anndata
scanpy
```

## 文件结构

```text
kpp_qvae/
├── models.py                  # QVAEEncoder / QVAEDecoder / CellQVAE
├── trainer.py                 # 数据准备、模型构建、训练循环、表征和 energy 提取
├── visualization.py           # 训练曲线、UMAP 和 energy 图
├── train_qvae_cell.py         # 命令行入口
├── evaluate_clustering.py      # Leiden 聚类指标评估
├── scripts/
│   ├── train.sh               # 当前训练命令
│   └── eval.sh                # 当前评估命令
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

## 聚类评估

```bash
bash scripts/eval.sh
```


评估脚本会扫描多个 Leiden resolution，并输出：

```text
ARI, AMI, NMI, Homogeneity, FMI, n_clusters
```

结果保存为：

```text
<output_dir>/X_qvae_clustering_metrics.csv
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
