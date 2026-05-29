# KPP QVAE Single-Cell Representation Learning

This directory provides an implementation of QVAE-based single-cell representation learning using `kaiwu-pytorch-plugin`. The workflow includes reading a single-cell expression matrix, training a QVAE, extracting low-dimensional representations, computing UMAP, analyzing the energy distribution, and evaluating clustering quality using cell-type labels.

## Data

The example dataset can be downloaded from:

https://www.kaggle.com/datasets/redwoodz/immune-data

## Dependencies

```bash
kaiwu==1.3.1
kaiwu-torch-plugin==0.1.0
torch==2.7.0
anndata
scanpy
leidenalg
scib_metrics
scgraph
```

## File Structure

```text
kpp_qvae/
├── models.py                  # QVAEEncoder / QVAEDecoder / CellQVAE
├── trainer.py                 # Data preparation, model construction, training loop, representation and energy extraction
├── visualization.py           # Training curves, UMAP, and energy plots
├── train_qvae_cell.py         # Command-line entry point
├── evaluate_clustering.py     # Legacy Leiden clustering evaluation entry point
├── evaluate_benchmark.py      # Unified evaluation entry point
├── scripts/
│   ├── train.sh               # Current training command
│   ├── eval_clustering.sh     # Clustering evaluation command
│   └── eval.sh                # Comprehensive evaluation command
└── README.md
```

## Training

The default data file is:

```text
../immune_processed.h5ad
```

The default observation fields are:

```text
batch_key = batch
labels_key = final_annotation
```

Run training:

```bash
bash scripts/train.sh
```

Quick check:

```bash
python train_qvae_cell.py \
  --epochs 1 \
  --sampler-type sa \
  --loss-type mse \
  --representation q \
  --output-dir ./outputs_smoke
```

Specify another dataset:

```bash
python train_qvae_cell.py \
  --data-path /path/to/data.h5ad \
  --epochs 100 \
  --output-dir ./outputs_custom
```

## Outputs

The training script will generate the following files under `--output-dir`:

```text
immune_kpp_qvae_best.pth          # Best model weights
model_epoch*.pth                  # Intermediate weights saved according to --checkpoint-every
X_qvae.npy                        # QVAE representation matrix
immune_kpp_qvae.h5ad              # AnnData file with X_qvae and QVAE_Energy written into it
training_history.csv              # Training metrics
training_curves.png               # Loss / reconstruction / KL curves
qvae_umap_labels_batches.png      # Cell-type and batch UMAP
qvae_energy_umap.png              # Energy UMAP
qvae_energy_by_celltype.png       # Energy distribution by cell type
```

## Evaluation

The unified evaluation entry point is `evaluate_benchmark.py`. Use `--metrics` to select the evaluation items to run:

| Evaluation Method | Purpose                                                                                                                                                                 |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| clustering        | Leiden clustering metrics: ARI, AMI, NMI, Homogeneity, FMI. Higher is better.                                                                                           |
| classification    | Downstream cell-type classification using Logistic Regression.                                                                                                          |
| scib              | scIB biological conservation and batch-correction metrics, assessing whether the representation preserves cell-type structure while removing unnecessary batch effects. |
| scgraph           | scGraph graph-structure evaluation.                                                                                                                                     |
| dpt               | DPT pseudotime and trajectory consistency, assessing whether the representation can produce a reasonable ordering of cell development or state transitions.             |
| all               | Run all of the above evaluation items.                                                                                                                                  |

Run clustering evaluation only:

```bash
python evaluate_benchmark.py \
  --h5ad ./outputs/immune_kpp_qvae.h5ad \
  --rep-key X_qvae \
  --label-key final_annotation \
  --metrics clustering
```

Run classification evaluation:

```bash
python evaluate_benchmark.py \
  --h5ad ./outputs/immune_kpp_qvae.h5ad \
  --rep-key X_qvae \
  --label-key final_annotation \
  --metrics classification
```

Run multiple evaluation items:

```bash
python evaluate_benchmark.py \
  --h5ad ./outputs/immune_kpp_qvae.h5ad \
  --rep-key X_qvae \
  --label-key final_annotation \
  --batch-key batch \
  --metrics clustering,classification,scib,scgraph
```

Run DPT:

```bash
python evaluate_benchmark.py \
  --h5ad ./outputs/immune_kpp_qvae.h5ad \
  --rep-key X_qvae \
  --label-key final_annotation \
  --metrics dpt \
  --root-cell-type HSPCs
```

`bash scripts/eval.sh` is equivalent to running `--metrics clustering`.

The results will be saved as corresponding CSV files and summarized into:

```text
<output_dir>/X_qvae_benchmark_summary.csv
```

## Key Parameters

```text
--latent-dim      Default: 256
--num-visible     Default: 128
--num-hidden      Default: 128
--sampler-type    sa or cim
--loss-type       Default: mse; options: mse or bernoulli
--representation  Default: q; options: zeta or q
--kl-beta         Default: 1e-5
--dist-beta       Default: 10.0
--lr              Default: 1e-4
--rbm-lr          Default: 1e-3
```
