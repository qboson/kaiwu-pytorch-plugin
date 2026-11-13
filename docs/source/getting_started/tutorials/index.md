<!-- ```YAML
title: Hands-on Tutorials — Quantum-accelerated Boltzmann Machines
slug: kpp-hands-on-tutorials
sidebar_position: 0
layout: home
hide: false
keywords:
  - KPP
  - Hands-on
``` -->

# Tutorials

> This chapter provides detailed hands-on tutorials to help you gain an in-depth understanding of various application scenarios of Kaiwu-PyTorch-Plugin.
>
> Before you start, make sure you have KPP installed and have gone through the [Quick Start](../quickstart.md). For theoretical background, see [Theoretical Foundations](../../theoretical-foundations/index.md).

## Architecture Overview

<!-- New to KPP? Start here to understand how the model families fit together. -->
New to KPP? Read this page before the hands-on tutorials. It complements the [Introduction](../introduction.md) system architecture from the model-family perspective.

```{toctree}
:maxdepth: 1
:hidden:

architecture_overview
```

- [KPP Architecture Overview](architecture_overview.md): Design philosophy, model hierarchy, and how RBM, DBN, Q-VAE, and Q-Diffusion build on each other.

## Hands-on Tutorials

Start from the sampling bottleneck and work through end-to-end applications that introduce one new concept at a time.

### Quantum Sampling for Energy-Based Models

```{toctree}
:maxdepth: 1
:hidden:

quantum_sampling_bottleneck
simulated_annealing
quantum_sampling_pytorch
```

- [Why Quantum? Revisiting the Sampling Bottleneck](quantum_sampling_bottleneck.md): The intractable partition function, slow MCMC mixing, and the quantum alternative.
- [Simulated Annealing for Ising Models](simulated_annealing.md): Mapping energy functions to Ising Hamiltonians and building a local sampling baseline.
- [Integrating Quantum Samplers into PyTorch](quantum_sampling_pytorch.md): Swapping the local SA sampler for the CIM sampler within the KPP training loop, with the core interfaces explained.

### Tutorial 1: Generative Modeling with a Full Boltzmann Machine

```{toctree}
:maxdepth: 1
:hidden:

bm_generation
```

- [BM Generation](bm_generation.md): Unsupervised data generation with a fully connected Boltzmann Machine. Example: `example/bm_generation/`.

### Tutorial 2: Feature Learning and Classification with RBMs and DBNs

```{toctree}
:maxdepth: 1
:hidden:

rbm_classification
dbn_classification
```

- [RBM Classification: Handwritten Digit Recognition](rbm_classification.md): Feature learning and classification with a single RBM. Example: `example/rbm_digits/rbm_digits.ipynb`.
- [DBN Classification: Deep Belief Networks](dbn_classification.md): Stacking RBMs into a Deep Belief Network for hierarchical features. Example: `example/dbn_digits/supervised_dbn_digits.ipynb`.

### Tutorial 3: Quantum Variational Autoencoder (Q-VAE)

```{toctree}
:maxdepth: 1
:hidden:

qvae_mnist
```

- [Q-VAE: Quantum Variational Autoencoder](qvae_mnist.md): Replacing the Gaussian prior with a quantum RBM for image generation and representation learning. Example: `example/qvae_mnist/run_pipeline.py`.

## Case Studies

Case studies build directly on the hands-on tutorials and show how KPP is used on larger-scale public datasets. Each study packages a complete pipeline: data loading, model definition, training, and evaluation, and lives alongside a runnable example under `example/`.

```{toctree}
:maxdepth: 1
:hidden:

qvae_cell
qvae_anomaly
qdiffusion
```

- [Q-VAE for Single-Cell Transcriptomics](qvae_cell.md): single-cell representation learning with a QVAE; UMAP and clustering evaluation (Leiden, classification, scIB, scGraph, DPT). Example: `example/qvae_cell/`.
- [Q-VAE for Energy-Supervised Anomaly Detection](qvae_anomaly.md): CleanEnergyQVAE with a free-energy scoring head and NCE-style contrastive objective on thyroid / creditcard benchmarks. Example: `example/qvae_anomaly/`.
- [Q-Diffusion for Protein Sequence Generation](qdiffusion.md): discrete diffusion with a quantum-accelerated energy reranker and a DPLM backbone. Example: `example/qdiffusion/`.

## Coming Soon

More tutorials and case studies are in progress:

- **Feature Selection with Quantum Annealing**: Using CIM for feature selection on tabular datasets, with linear regression and neural network downstream tasks. *Example: `example/feature_selection/`.*
- **Gaussian-Bernoulli RBM**: Extending RBM to continuous-valued inputs. *API: `GaussianBernoulliRestrictedBoltzmannMachine`.*
