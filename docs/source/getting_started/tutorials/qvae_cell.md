# Q-VAE for Single-Cell Transcriptomics

> This case study extends the [Q-VAE MNIST tutorial](qvae_mnist.md) to single-cell RNA sequencing (scRNA-seq) data. It demonstrates how Q-VAE learns low-dimensional representations of cell expression profiles, preserves cell-type structure, and corrects batch effects. Building on the Q-ELBO framework, this study adds clustering evaluation, UMAP visualization, and scIB benchmark metrics.

## Overview

By the end of this case study you will be able to:

- Apply Q-VAE to high-dimensional scRNA-seq expression matrices
- Extract low-dimensional representations that preserve cell-type identity
- Evaluate clustering quality with ARI, NMI, and scIB metrics
- Visualize representations with UMAP and energy distribution plots

## Running the Example

**Dependencies**:

```{code-block} bash
pip install kaiwu==1.3.1 scanpy anndata leidenalg scib_metrics
```

**Run**:

```bash
cd example/qvae_cell
bash scripts/train.sh
```

Or use the interactive notebook:

```bash
jupyter notebook train_qvae_cell.ipynb
```

**Key files**:

- `models.py`: `CellQVAE` — Q-VAE with batch-conditioned decoder for scRNA-seq
- `trainer.py`: `Trainer` — training loop with Q-ELBO optimization
- `evaluate_benchmark.py`: unified evaluation (clustering, classification, scIB, scGraph, DPT)
- `visualization.py`: UMAP plots, energy distribution, training curves

## How It Works

A Q-VAE for single-cell data replaces the Gaussian prior with a QBM, learning a low-dimensional representation that captures cell-type-specific expression patterns:

```
scRNA-seq (high-d)  →  Encoder  →  latent (low-d)  →  QBM prior  →  Decoder  →  reconstruction
```

The key adaptations from MNIST:

1. **Input dimension**: scRNA-seq matrices are typically 10k–30k genes, compared to 784 pixels in MNIST.
2. **Loss type**: MSE loss is used for continuous expression values (vs Bernoulli for binary pixels).
3. **Batch correction**: the Q-VAE representation learns to separate cell types while removing batch effects.
4. **Evaluation**: clustering metrics (ARI, NMI) and scIB benchmarks measure representation quality.

The training objective remains the Q-ELBO: reconstruction loss + KL divergence between encoder posterior and QBM prior. The sampler can be switched between SA and CIM without changing the rest of the pipeline.

---

## Quick Start

The steps below are the two command-line entry points of `example/qvae_cell`; `train_qvae_cell.ipynb` provides the interactive equivalent, and `scripts/train.sh` runs the first step (training) in one command with default settings.

**1. Train the Q-VAE.** `train_qvae_cell.py` trains `CellQVAE` on a scRNA-seq expression matrix and writes the learned representation back to an AnnData file. `--sampler-type` switches the QBM sampler between SA and CIM without touching the rest of the pipeline.

```bash
python train_qvae_cell.py \
  --dataset immune \
  --epochs 100 \
  --sampler-type sa \
  --loss-type mse \
  --output-dir ./outputs_immune_sa
```

**2. Run the evaluation benchmark.** `evaluate_benchmark.py` computes representation-quality metrics on the saved `X_qvae` embedding, grouped by `--metrics` (valid groups: `clustering`, `classification`, `scib`, `scgraph`, `dpt`, `all`). It prints a per-metric table and saves UMAP and energy-distribution plots.

```bash
python evaluate_benchmark.py \
  --h5ad ./outputs_immune_sa/immune_kpp_qvae.h5ad \
  --rep-key X_qvae \
  --label-key final_annotation \
  --metrics clustering,scib
```

The benchmark output is the result of this case study: clustering metrics (ARI, NMI) confirm that cell-type structure is preserved, while the scIB group reports batch-correction quality (e.g., batch ASW) — the numbers you compare across `--sampler-type sa` vs `cim` runs.

## Key Code

### 1. CellQVAE Model Architecture

The `CellQVAE` model extends the base Q-VAE with a batch-conditioned decoder
for scRNA-seq data:

```{literalinclude} ../../../../example/qvae_cell/models.py
:pyobject: CellQVAE
```

Key model details:
- Encoder: `QVAEEncoder` — maps scRNA-seq expression to latent space
- Decoder: `QVAEDecoder` — reconstructs expression from latent samples
- Batch conditioning: one-hot batch labels appended to decoder input
- QBM prior: trained with the same positive/negative phase as an RBM

### 2. Training Loop

The `Trainer` class implements the Q-ELBO optimization loop for scRNA-seq data:

```{literalinclude} ../../../../example/qvae_cell/trainer.py
:pyobject: Trainer.train
```

Key training details:
- Optimizer: Adam with learning rate scheduling
- Reconstruction loss: MSE (continuous expression values)
- KL term: pulls encoder posterior toward QBM prior
- Sampler: switch between SA and CIM via `--sampler-type`

### 3. Evaluation Benchmark

The `evaluate_benchmark.py` provides unified evaluation for clustering,
classification, and scIB metrics:

```{literalinclude} ../../../../example/qvae_cell/evaluate_benchmark.py
:pyobject: main
```

## Training and Evaluation

- **Training curves**: `plot_training_history` tracks reconstruction loss and KL divergence over epochs.
- **UMAP visualization**: `save_umap_plots` reduces the 256-dim latent representation to 2-D, colored by cell type.
- **Energy distribution**: `save_energy_plots` shows how QBM energy varies across cell types.
- **Clustering metrics**: ARI, AMI, NMI, Homogeneity, FMI measure clustering quality.
- **scIB benchmarks**: biological conservation and batch-correction metrics.

---

## Summary

- Q-VAE scales from MNIST to high-dimensional scRNA-seq data with minimal changes.
- The Q-ELBO framework remains the same; only the loss type and input dimension change.
- Learned representations preserve cell-type structure and correct batch effects.
- The sampler (SA vs CIM) can be switched without changing the rest of the pipeline.
