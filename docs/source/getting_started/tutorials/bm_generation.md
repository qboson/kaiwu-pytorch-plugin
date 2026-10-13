# BM Generation: Data Generation with a Fully Connected Boltzmann Machine

> This tutorial extends the RBM idea to a fully connected Boltzmann Machine (BM), where all nodes are connected (not just across layers). It demonstrates how to train a BM with a mixed KL-divergence + conditional likelihood objective and generate samples from it. The system architecture and sampling interface are introduced in [Introduction](../introduction.md) and [Quick Start](../quickstart.md); the theoretical background is covered in [Section 4.1: The Classic Boltzmann Machine](../../theoretical-foundations/kpp-theoretical-foundations-bm-overall.md). This tutorial focuses on what changes when moving from an RBM to a BM: the new objective terms and conditional sampling that a fully connected graph requires.

## Overview

By the end of this tutorial you will be able to:

- Understand the difference between an RBM (bipartite) and a BM (fully connected)
- Train a BM with a combined KL-divergence and negative conditional likelihood objective
- Implement learning-rate scheduling (cosine with warmup) and multi-process sampling
- Visualize generated sample distributions

## Running the Example

**Dependencies**:

```{code-block} bash
pip install kaiwu==1.3.1 pandas matplotlib
```

**Run**:

```bash
cd example/bm_generation
jupyter notebook train_bm.ipynb    # training
jupyter notebook sample_bm.ipynb  # sampling and evaluation
```

**Key files**:

- `trainer.py`: `Trainer` — BM training loop with cosine learning-rate schedule and multiprocess sampling
- `data_loader.py`: `CSVDataset` — CSV-to-PyTorch `Dataset` wrapper
- `saver.py`: `Saver` — checkpoint and loss-log persistence
- `train_bm.ipynb` / `sample_bm.ipynb`: interactive walkthroughs

## How It Works

Unlike an RBM, a BM has connections within the visible layer, so hidden and visible units cannot be sampled in parallel. The key differences are:

| Aspect | Restricted Boltzmann Machine (RBM) | Boltzmann Machine (BM) |
|---|---|---|
| Connections | Inter-layer only, no intra-layer edges | All nodes connected |
| Sampling | Parallel (conditional independence within layers) | Sequential (Gibbs sampling or solved by an external sampler) |
| KPP method | `rbm.get_hidden(v)` | `bm.condition_sample(sampler, v)` |
| Training efficiency | Higher | Lower |
| Expressive power | Limited by bipartite structure | Stronger; can model arbitrary distributions |

This example is suitable for:

- Modeling complex dependencies between variables that an RBM's bipartite structure cannot capture
- Small-scale sample generation where fully connected interactions matter
- Studying Boltzmann distribution sampling behavior

The training objective combines two terms:

1. **KL divergence**: encourages the model distribution to match the data distribution (unsupervised generation).
2. **Negative conditional likelihood (NCL)**: when part of the visible units are designated as outputs (`num_output`), the model must reconstruct those outputs given the rest.

The combined objective is:

$$
\mathcal{L} = \alpha \cdot \text{KL\_divergence} + (1 - \alpha) \cdot \text{NCL}
$$

where $\alpha$ is set via `cost_param["alpha"]`.

The `num_output` parameter designates which visible units are "outputs" (e.g. labels in a supervised setting), enabling conditional generation — you can fix part of the visible units and let the model generate the rest.

### Sampling Phases

**Negative phase**: `bm_net.sample(worker)` draws a complete joint visible+hidden state from the current BM distribution, approximating the model distribution for the contrastive divergence objective.

**Positive phase**: the positive phase uses two conditional samplings per batch:

- **Full conditional**: the entire input batch is fixed as the visible layer, and the corresponding joint state is sampled.
- **Partial conditional**: only the non-output portion of the input (`data[:, :-num_output]`) is fixed, and the output dimensions are sampled freely.

These two samplings compute the KL term and the NCL term respectively. A cosine schedule with linear warmup stabilizes early training. Multi-process sampling splits each batch across workers, running positive-phase conditional sampling in parallel to accelerate the training loop. All sampling operations are embedded in the PyTorch computation graph, enabling end-to-end backpropagation.

## Quick Start

The following snippet shows how to run the training pipeline from a user's perspective, this is exactly what you would execute in `train_bm.ipynb`:

Initialize the BM trainer:

```python
from trainer import Trainer
from data_loader import CSVDataset
from saver import Saver

dataset = CSVDataset("data.csv")
trainer = Trainer(
    n_visible=128, n_hidden=64, num_output=2,
    alpha=0.5, beta=1e-4, lr=1e-3,
)
```

Train with multiprocess sampling:

```python
trainer.train(dataset, epochs=50, batch_size=32, n_workers=4)
```

After training, generate samples from `sample_bm.ipynb`: the BM samples full visible+hidden states, and the visible portion is rendered as the generated data distribution.

## Key Code

### 1. Trainer Initialization

The `Trainer` class initializes a fully connected `BoltzmannMachine` and sets up shared memory for multiprocess sampling:

```{literalinclude} ../../../../example/bm_generation/trainer.py
:pyobject: Trainer.__init__
```

Key initialization details:
- `BoltzmannMachine(num_nodes=num_visible + num_hidden)` — fully connected graph
- `share_memory()` — enables multi-process sampling without copying
- `cost_param = {"alpha": 0.5, "beta": 0.5}` — KL/NCL balance

---

### 2. Training Loop with Cosine Schedule

The training loop uses Adam optimizer with cosine learning-rate schedule and multi-process conditional sampling:

```{literalinclude} ../../../../example/bm_generation/trainer.py
:pyobject: Trainer.train
```

Key training details:
- Optimizer: Adam with weight decay
- Learning rate: cosine schedule with linear warmup (`num_warmup_steps = max_steps / 20`)
- Multi-process: `num_processes` workers run `condition_sample` in parallel

---

### 3. Multi-process Sampling Worker

The worker function runs in separate processes, performing both full and partial conditional sampling for the KL and NCL terms:

```{literalinclude} ../../../../example/bm_generation/trainer.py
:pyobject: process_solve_graph_worker
```

## Training and Evaluation

- **Loss curve**: `Saver` appends each step's loss to `loss.txt`; plot it to confirm convergence.
- **Checkpoints**: the model is saved periodically as `checkpoint_step_<n>.pt` for later sampling or resumption.
- **Weight visualization**: during training, the trainer can render weight matrices and their gradients for debugging.
- **Generated samples**: compare the distribution of generated samples against the original data (histograms / scatter plots in `sample_bm.ipynb`).

## Summary

- A BM drops the bipartite restriction, enabling arbitrary dependencies but requiring sequential sampling.
- The training objective mixes KL divergence (generation) and NCL (conditional reconstruction), controlled by `alpha`.
- The `num_output` parameter designates output units, enabling conditional generation.
- Multi-process sampling accelerates the negative phase; cosine warmup stabilizes early training.
