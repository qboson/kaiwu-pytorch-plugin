# Q-VAE for Energy-Supervised Anomaly Detection

> This case study extends the [Q-VAE MNIST tutorial](qvae_mnist.md) to tabular anomaly detection. It introduces CleanEnergyQVAE: a Q-VAE with a free-energy scoring head, where normal samples get lower free energy and anomalies get higher free energy. Built on the Q-ELBO framework, this study adds threshold search and PR/ROC evaluation.

## Overview

By the end of this case study you will be able to:

- Extend Q-VAE with a free-energy scoring head for anomaly detection
- Train with energy-supervised loss to separate normal from anomalous samples
- Search for optimal threshold using best-F1 over a candidate grid
- Evaluate with PR-AUC, ROC-AUC, and score distribution plots

## Running the Example

**Dependencies**:

```{code-block} bash
pip install kaiwu scikit-learn matplotlib seaborn tqdm
```

**Run**:

```bash
cd example/qvae_anomaly
bash run_public_ad.sh
```

Or use the interactive notebook:

```bash
jupyter notebook qvae_anomaly_demo.ipynb
```

**Key files**:

- `model/model.py`: `CleanEnergyQVAE` — Q-VAE with free-energy scoring head
- `model/losses.py`: energy-supervised loss functions (margin / InfoNCE)
- `trainer/trainer.py`: N-epoch training loop with tqdm
- `utils/evaluate.py`: `optimize_threshold_rbm` — best-F1 threshold search
- `utils/visualize.py`: score distribution, PR/ROC curves, t-SNE

## How It Works

CleanEnergyQVAE extends the standard Q-VAE with an RBM free-energy scoring head:

```
tabular data  →  Encoder  →  latent  →  QBM prior  →  Decoder  →  reconstruction
                            ↓
                      free-energy score (normal = low, anomaly = high)
```

The key innovations:

1. **Free-energy scoring head**: an RBM on top of the latent space computes free energy as the anomaly score.
2. **Energy-supervised loss**: normal samples are pulled toward the QBM prior, anomalies are pushed away.
3. **Threshold search**: best-F1 over a candidate grid determines the decision boundary.
4. **Evaluation**: PR-AUC, ROC-AUC, precision, recall, F1 measure anomaly detection quality.

The Q-ELBO still provides the reconstruction loss and KL divergence; the free-energy head adds an anomaly-specific scoring objective.

---

## Quick Start

The steps below are the two command-line entry points of `example/qvae_anomaly`; `qvae_anomaly_demo.ipynb` provides the interactive equivalent, and `run_public_ad.sh` runs both steps in one command.

**1. Train.** `train_public_ad.py` trains `CleanEnergyQVAE` with the energy-supervised loss on a public anomaly benchmark. `--dataset` selects the tabular benchmark (e.g., `thyroid`, `creditcard`); `--lambda-anom` sets the weight of the anomaly energy term relative to the Q-ELBO objective.

```bash
python train_public_ad.py --dataset thyroid --epochs 20 --lambda-anom 1.0
```

**2. Evaluate.** `evaluate_public_ad.py` loads the best checkpoint, searches the best-F1 threshold over a candidate grid, and reports PR-AUC / ROC-AUC with precision, recall, and F1. It also saves the score distribution and PR/ROC curves under the output directory.

```bash
python evaluate_public_ad.py --dataset thyroid
```

The result is the anomaly-detection report: a clear low/high free-energy separation of normal vs anomalous samples, a best-F1 decision threshold, and the PR-AUC / ROC-AUC you compare across `--lambda-anom` and `--sampler-type` runs.

## Key Code

### 1. CleanEnergyQVAE Model

The `CleanEnergyQVAE` model combines a standard Q-VAE with an RBM free-energy
scoring head:

```{literalinclude} ../../../../example/qvae_anomaly/model/model.py
:pyobject: CleanEnergyQVAE
```

Key model details:
- Encoder/Decoder: `CleanEncoder` / `CleanDecoder` — standard Q-VAE architecture
- Free-energy head: RBM on latent space for anomaly scoring
- Anomaly score: free energy (normal = low, anomaly = high)
- Loss types: margin, InfoNCE, or both (configurable)

### 2. Energy-Supervised Loss

The `losses.py` implements energy-supervised loss to separate normal from
anomalous samples:

```{literalinclude} ../../../../example/qvae_anomaly/model/losses.py
:pyobject: _compose_energy_supervised_qvae_loss_terms
```

Key loss details:
- Normal samples: pulled toward QBM prior (low free energy)
- Anomalies: pushed away from QBM prior (high free energy)
- `lambda_anom`: weight for anomaly energy loss (default 1.0)

### 3. Threshold Search and Evaluation

The `optimize_threshold_rbm` function implements best-F1 threshold search
over a candidate grid:

```{literalinclude} ../../../../example/qvae_anomaly/utils/evaluate.py
:pyobject: optimize_threshold_rbm
```

## Training and Evaluation

- **Score distribution**: `plot_score_distribution` shows free energy by label (normal vs anomaly).
- **PR/ROC curves**: `plot_pr_roc_curve` generates precision-recall and receiver operating characteristic curves.
- **Latent t-SNE**: `plot_latent_tsne` reduces latent space to 2-D, colored by label.
- **Timeseries**: `plot_timeseries_anomaly` shows energy vs time for temporal anomaly detection.
- **Metrics**: PR-AUC, ROC-AUC, precision, recall, F1.

---

## Summary

- CleanEnergyQVAE extends Q-VAE with a free-energy scoring head for anomaly detection.
- Energy-supervised loss separates normal from anomalous samples in latent space.
- Threshold search with best-F1 determines the decision boundary.
- The Q-ELBO framework remains the same; only the scoring head and loss objective are added.
