# QVAE-Anomaly: Clean Energy QVAE for Anomaly Detection

This directory provides an implementation of **CleanEnergyQVAE** for public anomaly detection benchmarks. The model combines a Quantum Variational Autoencoder (QVAE) with an RBM free-energy scoring head + NCE contrastive loss to separate normal from anomalous samples.

## Datasets

Download and place the datasets under `data/`:

| Dataset | File | Source |
|---|---|---|
| Thyroid | `data/38_thyroid.npz` | [ADBench](https://github.com/Minqi824/ADBench) |
| Creditcard | `data/creditcard.csv` | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

## Dependencies

```bash
kaiwu>=1.3.1
torch
scikit-learn
numpy
pandas
matplotlib
seaborn
tqdm
```

## File Structure

```text
qvae_anomaly/
├── model/                  # Model definitions
│   ├── config.py           # Config dataclass (auto device + field validation)
│   ├── model.py            # CleanEnergyQVAE + build_model() factory
│   ├── losses.py           # InfoNCE + reconstruction losses
│   └── networks.py         # CleanEncoder / CleanDecoder / ResidualMLPBlock
├── trainer/                # Training loop
│   ├── tuner.py            # Single-epoch train/eval
│   └── trainer.py          # N-epoch loop with tqdm + logging
├── utils/                  # Utilities
│   ├── datasets.py         # Unified dataset loader (thyroid, creditcard)
│   ├── evaluate.py         # optimize_threshold_rbm (best-F1 threshold search)
│   ├── visualize.py        # Plots (score distribution, PR/ROC, t-SNE, timeseries)
│   ├── logging.py          # get_logger with file + stdout
│   └── exception.py        # Exception hierarchy
├── data/                    # Datasets (user-provided, not in repo)
├── outputs/                 # Checkpoints + results (per-dataset subdirs)
├── tests/                   # Unit tests
│   ├── smoke.py             # 1-epoch smoke test
│   ├── test_model.py        # Model build / forward / score
│   ├── test_losses.py       # Loss function tests
│   └── test_datasets.py     # Dataset loading tests
├── train_public_ad.py       # Training entry point
├── evaluate_public_ad.py    # Evaluation entry point
├── run_public_ad.sh         # One-click train + evaluate
├── qvae_anomaly_demo.ipynb  # Interactive notebook (train + evaluate + inline plots)
└── README.md
```

## Quick Start

### Install dependencies

```bash
pip install kaiwu scikit-learn matplotlib seaborn tqdm
```

### Prepare data

Place `38_thyroid.npz` and `creditcard.csv` under `data/`.

### One-click run (thyroid, 20 epochs)

```bash
bash run_public_ad.sh
```

### Customize dataset / epochs

```bash
# Creditcard, 20 epochs
DATASET=creditcard bash run_public_ad.sh

# Quick test (5 epochs)
EPOCHS=5 bash run_public_ad.sh

# Both
DATASET=creditcard EPOCHS=10 bash run_public_ad.sh
```

### Run step by step

```bash
# Train
python train_public_ad.py --dataset thyroid --epochs 20

# Evaluate (uses best checkpoint)
python evaluate_public_ad.py --dataset thyroid
```

### Interactive notebook

```bash
jupyter notebook qvae_anomaly_demo.ipynb
```

Edit the `DATASET` cell (`'thyroid'` or `'creditcard'`) and run all. Metrics are shown as a table and plots render inline. Notebook outputs go to `outputs/notebook/<dataset>/`, kept separate from the script runs in `outputs/<dataset>/`.

## Outputs

Results are written to `outputs/<dataset>/`:

```text
outputs/
├── thyroid/
│   ├── train.log                # Training log (per-epoch metrics)
│   ├── evaluate.log              # Evaluation log
│   ├── summary.json              # PR-AUC / ROC-AUC / Precision / Recall / F1
│   ├── thyroid.pt                # Best model checkpoint
│   ├── score_distribution.png    # Energy distribution by label
│   ├── pr_roc_curve.png          # PR + ROC curves
│   ├── latent_tsne.png           # Latent space t-SNE (creditcard only)
│   └── timeseries_anomaly.png    # Energy vs time (creditcard only)
└── creditcard/
    └── ...
```

## Key Parameters

```text
--dataset        thyroid or creditcard
--epochs         Default: 20
--lr             Default: 5e-4
--batch-size     Default: 512
--lambda-anom    Default: 50.0  (anomaly energy loss weight)
--kl-beta        Default: 1e-4  (KL annealing weight)
--hidden         Default: auto by dataset
--rbm            Default: auto by dataset
```

Architecture defaults per dataset:

| Dataset | hidden | RBM |
|---|---|---|
| thyroid | 32 | 16x16 |
| creditcard | 128 | 32x32 |

## Tests

```bash
# Smoke test (1 epoch, dummy data)
python tests/smoke.py

# Model tests
python tests/test_model.py

# Loss tests
python tests/test_losses.py

# Dataset tests
python tests/test_datasets.py thyroid
python tests/test_datasets.py creditcard
```
