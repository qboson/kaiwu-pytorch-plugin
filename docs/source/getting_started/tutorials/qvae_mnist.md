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

## Key Code

Load MNIST and build the Q-VAE pipeline:

```python
from utils.loadMNIST import loadMNIST
from run_pipeline import get_full_pipeline

train_loader, test_loader = loadMNIST(dataset="mnist", batch_size=128, flatten=True)
model, classifier = get_full_pipeline(latent_dim=64, sampler_type="cim")
```

Train the Q-VAE:

```python
from trainer.trainer import Trainer

trainer = Trainer(model, train_loader, test_loader, epochs=50, lr=1e-3)
trainer.train()
```

Evaluate downstream classification on learned representations:

```python
from downstream.classifier import MLPClassifier

clf = MLPClassifier(hidden_dim=128)
clf.fit(model, train_loader, test_loader, epochs=30)
```

## Training and Evaluation

- **Training curves**: `plot_training_curves` tracks reconstruction loss, KL divergence, and validation accuracy over epochs.
- **Latent space**: `t_SNE` reduces the 64-dim latent representation to 2-D, colored by digit class, revealing cluster structure.
- **Generated samples**: after training, the decoder can generate new images by sampling from the QBM prior and decoding.

## Case Studies

The Q-VAE architecture introduced here scales to larger biological and tabular datasets. See the [Case Studies section](index.md) for end-to-end pipelines built on this tutorial:

- [Q-VAE for Single-Cell Transcriptomics](qvae_cell.md): representation learning on scRNA-seq with UMAP and clustering evaluation.
- [Q-VAE for Energy-Supervised Anomaly Detection](qvae_anomaly.md): CleanEnergyQVAE with a free-energy scoring head on tabular benchmarks.

## Summary

- A Q-VAE swaps the Gaussian prior for a QBM, trained with the same positive/negative phase as an RBM.
- The Q-ELBO combines reconstruction loss with a KL term between the encoder posterior and the QBM prior.
- The sampler (SA vs CIM) can be switched without changing the rest of the pipeline.
- Learned representations can be fed to a downstream classifier, as in the RBM feature-learning tutorial.
