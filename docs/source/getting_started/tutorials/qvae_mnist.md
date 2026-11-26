# Q-VAE: Quantum Variational Autoencoder on MNIST

> This tutorial combines [Section 4.2: Restricted Boltzmann Machine](../../theoretical-foundations/kpp-theoretical-foundations-bm-restricted.md) with the VAE framework. It replaces the Gaussian prior of a standard VAE with a quantum Boltzmann machine (QBM) and trains the resulting Q-VAE on MNIST for both image generation and representation learning. Building on the [Quick Start](../quickstart.md) training loop, this tutorial adds the encoder–decoder structure and the KL term between the encoder posterior and the QBM prior.

## Overview

By the end of this tutorial you will be able to:

- Understand how a QBM prior replaces the Gaussian prior in a VAE
- Train a Q-VAE on MNIST and generate reconstructed images
- Extract learned representations and evaluate them on a downstream classification task
- Visualize the latent space with t-SNE

## Running the Example

**Dependencies**:

```{code-block} bash
pip install torchvision==0.22.0 torchmetrics[image]
```

**Run**:

```bash
cd example/qvae_mnist
python run_pipeline.py --epochs 50 --run-tsne
```

Or use the interactive notebooks:

```bash
jupyter notebook train_qvae.ipynb            # train and generate
jupyter notebook train_qvae_classifier.ipynb  # representation learning + classification
```

**Key files**:

- `run_pipeline.py`: end-to-end entry (load data → train Q-VAE → optional t-SNE)
- `model/networks.py`: `BasicEncoder` / `BasicDecoder`
- `trainer/trainer.py`: `Trainer.train` — Q-ELBO optimization loop
- `downstream/classifier.py`: `MLPClassifier.fit` — downstream classifier on learned representations

## How It Works

A standard VAE uses a Gaussian prior $p(\mathbf{z})$ and an encoder $q_\phi(\mathbf{z}|\mathbf{x})$. A Q-VAE replaces the Gaussian prior with a QBM:

```
x (image)  →  Encoder  →  z (latent)  →  QBM prior p(z)  →  Decoder  →  x_recon
```

The three components are:

- **Encoder**: maps the input image $\mathbf{x}$ to the approximate posterior $q_\phi(\mathbf{z}|\mathbf{x})$, parameterized by a neural network.
- **QBM prior**: models the latent distribution $p_\theta(\mathbf{z})$ using a quantum Boltzmann machine instead of a Gaussian.
- **Decoder**: maps the latent sample $\mathbf{z}$ back to data space, reconstructing the input as a Bernoulli distribution.

The training objective is the quantum evidence lower bound (Q-ELBO):

$$
\mathcal{L}_{\text{Q-ELBO}} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [\log p_\theta(\mathbf{x}|\mathbf{z})] - \tilde{H}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}))
$$

The first term is the reconstruction loss: how well the decoder reconstructs the input from the latent sample. The second term is a KL-like divergence that pulls the encoder posterior $q_\phi(\mathbf{z}|\mathbf{x})$ toward the QBM prior $p_\theta(\mathbf{z})$.

The QBM prior is trained using the same positive/negative phase as an RBM: positive phase from the encoder samples, negative phase from QBM sampling. The sampler can be switched between `sa` (classical simulated annealing) and `cim` (quantum CIM) via the `--sampler-type` flag — the rest of the pipeline is unchanged.

---

## Quick Start

The steps below condense `example/qvae_mnist/train_qvae.ipynb` with imports omitted; the notebook is the interactive walkthrough, and `python run_pipeline.py --epochs 50 --run-tsne` is the equivalent end-to-end script (it also trains the downstream classifier).

**1. Build the config.** The `Config` object holds the dataset name, data path, model type, and training hyperparameters in one place.

```python
config = Config(
    model_type="QVAE",
    name="mnist",               # "mnist" / "fashion-mnist" / "kmnist"
    data_path="./data",
    batch_size=256,
    num_epochs=10,
    lr=1e-3,
    num_train_samples=60000,
    num_test_samples=10000,
    use_cuda=False,
)
```

**2. Train the Q-VAE.** `Trainer(config=config)` builds the model from the config; `train` returns the trained model and the per-epoch train/test losses. With `run_tsne=True` it additionally projects the 64-d latent space to 2-D (colored by digit) and records a training animation. The t-SNE plot is the first check that the QBM prior has learned a structured latent space.

```python
trainer = Trainer(config=config)
qvae_model, train_losses, test_losses = trainer.train(
    run_tsne=True, generate_animation=True, tsne_interval=5,
)
```

**3. Inspect the result.** A steadily decreasing Q-ELBO loss (reconstruction + KL) and a t-SNE with digit-separated clusters confirm training succeeded.

```python
print(f"final train loss: {train_losses[-1]:.4f} | test loss: {test_losses[-1]:.4f}")
```

**Evaluation beyond training.** Two downstream tasks complete the picture and are covered in dedicated sections instead of here:
- **Representation learning + classification**: train a classifier on the learned features (`train_qvae_classifier.ipynb`, or the end-to-end CLI `python run_pipeline.py --epochs 50 --run-tsne`), then report test accuracy — see *Key Code → §3 Downstream Classification on Learned Representations*.
- **Generation + FID**: `train_qvae.ipynb` samples new images from the QBM prior (`generate_qvae_images`, saved as a grid) and scores them against 10,000 real test images with FID — see *Training and Evaluation* below.

## Key Code

### 1. Model Configuration and Encoder/Decoder

The Q-VAE model uses a standard encoder-decoder architecture, with the QBM prior replacing the Gaussian prior in the latent space:

```{literalinclude} ../../../../example/qvae_mnist/model/networks.py
:pyobject: BasicEncoder
```

Key model details:
- Encoder: maps 784-d MNIST images to 64-d latent space
- Decoder: reconstructs images from latent samples
- QBM prior: trained with the same positive/negative phase as an RBM

### 2. Q-ELBO Training Loop

The `Trainer` class implements the Q-ELBO optimization loop, combining reconstruction loss with the KL term between encoder posterior and QBM prior:

```{literalinclude} ../../../../example/qvae_mnist/trainer/trainer.py
:pyobject: Trainer.train
```

Key training details:
- Optimizer: Adam with learning rate and weight decay from config
- Reconstruction loss: Bernoulli cross-entropy (or MSE for continuous data)
- KL term: pulls encoder posterior toward QBM prior
- Sampler: switch between SA and CIM via `--sampler-type` flag

### 3. Downstream Classification on Learned Representations

After training the Q-VAE, the encoder's hidden representations can be fed to a downstream classifier for supervised evaluation:

```{literalinclude} ../../../../example/qvae_mnist/downstream/classifier.py
:pyobject: MLPClassifier.fit
```

## Training and Evaluation

- **Training curves**: `plot_training_curves` tracks reconstruction loss, KL divergence, and validation accuracy over epochs.
- **Latent space**: `t_SNE` reduces the 64-dim latent representation to 2-D, colored by digit class, revealing cluster structure.
- **Generated samples**: after training, the decoder can generate new images by sampling from the QBM prior and decoding.

## Case Studies

The Q-VAE architecture introduced here scales to larger biological and tabular datasets. See the [Case Studies section](index.md) for end-to-end pipelines built on this tutorial:

- [Q-VAE for Single-Cell Transcriptomics](qvae_cell.md): representation learning on scRNA-seq with UMAP and clustering evaluation.
- [Q-VAE for Energy-Supervised Anomaly Detection](qvae_anomaly.md): CleanEnergyQVAE with a free-energy scoring head on tabular benchmarks.

---

## Summary

- A Q-VAE swaps the Gaussian prior for a QBM, trained with the same positive/negative phase as an RBM.
- The Q-ELBO combines reconstruction loss with a KL term between the encoder posterior and the QBM prior.
- The sampler (SA vs CIM) can be switched without changing the rest of the pipeline.
- Learned representations can be fed to a downstream classifier, as in the RBM feature-learning tutorial.
