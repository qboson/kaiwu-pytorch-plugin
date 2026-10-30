# KPP Architecture Overview

> This page explains how KPP's model components fit together and how different model families (RBM, DBN, Q-VAE, Q-Diffusion) build on top of each other. For the high-level system architecture and component interaction diagram, see the [Introduction](../introduction.md).

## Design Philosophy

KPP is built on two core principles:

1. **Decoupled architecture**: energy model definition is separated from sampling execution. You can swap samplers (SA ↔ CIM) without changing the model code.
2. **Composition over inheritance**: base energy models (RBM/BM) are combined with user-provided components (encoder, decoder) to build application-specific models.

This means you can start with a simple RBM for feature extraction, then extend it into a DBN for deeper representations, or wrap it in a Q-VAE framework for generative modeling — all within the same PyTorch workflow.

---

## Model Families

Right now, KPP provides four main model families, each building on the previous one.

> 📖 **Complete API reference**: See the full [Module API](../../modules/kaiwu.torch_plugin.md) for all classes, methods, and parameters.

### 1. Boltzmann Machine (BM)

**The foundation**: an energy-based model family with visible and hidden units, binary by default, with a Gaussian-Bernoulli variant that models one side as Gaussian for real-valued inputs.

- **Full Boltzmann Machine**: fully connected graph, all units can connect to each other. Requires sequential sampling.
- **Restricted Boltzmann Machine (RBM)**: bipartite graph, visible units only connect to hidden units, no visible-visible or hidden-hidden connections. This enables parallel sampling within layers.
- **Gaussian-Bernoulli RBM (GBRBM)**: extends RBM to continuous-valued inputs, one partition is Gaussian (with learned mean/variance) and the other stays Bernoulli; `is_visible_gaussian` picks which side is Gaussian (default: visible).

**Main classes**: `BoltzmannMachine`, `RestrictedBoltzmannMachine`, `GaussianBernoulliRestrictedBoltzmannMachine`

**Typical use**: unsupervised feature learning, generative modeling.

→ Hands-on: [RBM Classification](rbm_classification.md), [BM Generation](bm_generation.md), GBRBM (Coming Soon)

---

### 2. Deep Belief Network (DBN)

**Extension**: stack multiple RBMs into a deep network.

Each RBM is trained greedily layer-by-layer (unsupervised), then the whole stack can be fine-tuned (supervised).

```
Input → RBM1 → features1
              ↓
            RBM2 → features2
                  ↓
                RBM3 → features3
                        ↓
                      classifier
```

**Main class**: `UnsupervisedDBN`

**Typical use**: hierarchical feature learning, deep representations.

→ Hands-on: [DBN Classification](dbn_classification.md)

---

### 3. Quantum Variational Autoencoder (Q-VAE)

**Extension**: combine an encoder-decoder architecture with a QBM prior.

Instead of the standard Gaussian prior in a VAE, Q-VAE uses a **quantum Boltzmann machine (QBM)**, a Boltzmann machine whose negative-phase sampling is handled by the SA/CIM sampler, as the latent prior.

```
Input → Encoder → latent z → QBM prior → Decoder → reconstruction
```

**Components**:
- Encoder: maps input to a latent distribution
- Decoder: reconstructs input from latent samples
- QBM prior: a Boltzmann machine prior sampled by SA or CIM
- Sampler: SA (classical) or CIM (quantum)

**Main classes**: `QVAE`, `AutoEncoderBase`

**Typical use**: generative modeling, representation learning, anomaly detection.

**Application variants** (in `example/`):
- `MNISTQVAE`: image generation (784-d input)
- `CellQVAE`: single-cell transcriptomics (10k+ genes, batch-conditioned)
- `CleanEnergyQVAE`: anomaly detection (free-energy scoring head)

→ Hands-on: [Q-VAE MNIST](qvae_mnist.md), [Q-VAE Cell](qvae_cell.md), [Q-VAE Anomaly](qvae_anomaly.md)

---

### 4. Q-Diffusion

**Extension**: discrete diffusion model with a BM energy reranker.

Combines a pre-trained protein language model (DPLM) as the proposal model with a trainable BM energy reranker to improve generated sequence quality.

```
Protein sequence → DPLM proposal → logits → sample candidates
                                                    ↓
                                              BM energy reranker
                                                    ↓
                                          energy-weighted output
```

**Main classes**: `QDiffusion`, `EnergyModel`, `QDiffusionConfig`

**Typical use**: protein sequence generation, energy-guided sampling.

**Application** (in `example/`):
- `BMConditionedEnergyModel`: DPLM feature encoder + BM visible-space reranker

→ Hands-on: [Q-Diffusion](qdiffusion.md)

---

## How to Build a Custom Model

### Step 1: Choose your base
- Need feature extraction? → Start with [RBM](rbm_classification.md)
- Need generative modeling? → Start with [BM](bm_generation.md)
- Need representation learning? → Start with [Q-VAE](qvae_mnist.md)

### Step 2: Provide your components
```python
from kaiwu.torch_plugin import QVAE
import torch.nn as nn

# Define your encoder/decoder
encoder = nn.Sequential(...)
decoder = nn.Sequential(...)

# Build the model
model = QVAE(
    input_dimension=784,
    encoder=encoder,
    decoder=decoder,
    sampler_type="sa",  # or "cim"
)
```

### Step 3: Train with the standard PyTorch loop
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = model.training_step(batch)
        loss.backward()
        optimizer.step()
```

---

## Cross-reference Map

| Topic | See also |
|---|---|
| **Complete API reference** | [Module API](../../modules/kaiwu.torch_plugin.md) |
| System architecture overview | [Introduction](../introduction.md) |
| Mathematical foundations | [Theoretical Foundations](../../theoretical-foundations/index.md) |
| RBM / BM hands-on | [RBM Classification](rbm_classification.md), [BM Generation](bm_generation.md) |
| DBN hands-on | [DBN Classification](dbn_classification.md) |
| Q-VAE hands-on & case studies | [Q-VAE MNIST](qvae_mnist.md), [Q-VAE Cell](qvae_cell.md), [Q-VAE Anomaly](qvae_anomaly.md) |
| Q-Diffusion hands-on | [Q-Diffusion](qdiffusion.md) |
