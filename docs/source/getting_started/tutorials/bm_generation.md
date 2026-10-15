# BM Generation: Data Generation with a Fully Connected Boltzmann Machine

> This tutorial extends the RBM idea to a fully connected Boltzmann Machine (BM), where all nodes are connected (not just across layers). It demonstrates how to train a BM with a mixed KL-divergence + non-contrastive loss (NCL) objective and generate samples from it. The system architecture and sampling interface are introduced in [Introduction](../introduction.md) and [Quick Start](../quickstart.md); the theoretical background is covered in [Section 4.1: The Classic Boltzmann Machine](../../theoretical-foundations/kpp-theoretical-foundations-bm-overall.md). This tutorial focuses on what changes when moving from an RBM to a BM: the new objective terms and conditional sampling that a fully connected graph requires.

## Overview

By the end of this tutorial you will be able to:

- Understand the difference between an RBM (bipartite) and a BM (fully connected)
- Train a BM with a combined KL-divergence and non-contrastive loss (NCL) objective
- Implement learning-rate scheduling (cosine with warmup) and multi-process sampling
- Visualize generated sample distributions

The concrete example behind this page is small-sample **RNA-sequence data augmentation**: the BM is trained on 100 binarized 128-dimensional latent codes `z` from the `Aptaswitch` task, generates new codes, and generation quality is judged by a downstream discriminator's predictions before and after augmentation. The training mechanics below apply to any BM task.

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
2. **Non-contrastive loss (NCL)**: a conditional-consistency term that does not require the free negative-phase sampling. When part of the visible units are designated as outputs (`num_output`), the model must reconstruct those outputs given the rest.

The combined objective is:

$$
\mathcal{L} = \alpha \cdot \text{KL\_divergence} + (1 - \alpha) \cdot \text{NCL}
$$

where $\alpha$ is set via `cost_param["alpha"]`.

The `num_output` parameter designates which visible units are "outputs" (e.g. labels in a supervised setting), enabling conditional generation — you can fix part of the visible units and let the model generate the rest.


### Sampling Phases

**Negative phase**: `bm_net.sample(worker)` draws a complete joint visible+hidden state from the current BM distribution, approximating the model distribution for the contrastive divergence objective.

**Positive phase**: two conditional samplings per batch:

- **Full conditional**: the entire input batch is fixed as the visible layer, and the corresponding joint state is sampled.
- **Partial conditional**: only the non-output portion of the input (`data[:, :-num_output]`) is fixed, and the output dimensions are sampled freely.

Together they provide the positive-phase statistics. The KL term contrasts these with the negative-phase free sample (`state_all`). The NCL term instead measures an energy consistency between the two positive-phase states, with the output columns zeroed — it needs no negative-phase sampling.

All sampling runs on multiple workers: each batch is split across processes, so the two conditional samplings proceed in parallel. A cosine schedule with linear warmup stabilizes early training. All sampling operations stay inside the PyTorch computation graph, enabling end-to-end backpropagation.

The sampler abstraction used here, i.e., swap SA ↔ CIM without touching the rest of the pipeline, is introduced in [Integrating Quantum Samplers into PyTorch](quantum_sampling_pytorch.md).

## Quick Start

The steps below condense `example/bm_generation/train_bm.ipynb` with imports omitted; the notebook is the fully runnable walkthrough.

**1. Build the sampler.** Use simulated annealing (SA) for local debugging; for quantum hardware, swap in a `CIMOptimizer` wrapped in `PrecisionReducer`, the rest of the pipeline stays unchanged.

```python
sampler = kw.classical.SimulatedAnnealingOptimizer(
    initial_temperature=100, alpha=0.9, cutoff_temperature=1e-2,
    iterations_per_t=200, size_limit=10, rand_seed=1234,
)
```

**2. Wrap the data.** `CSVDataset` takes a numpy array shaped `(num_columns, num_rows)` and `DataLoader` provides batching.

```python
data = CSVDataset(np.load("./data_input/z_128_100.npy").T)
dataloader = torch.utils.data.DataLoader(data, batch_size=15, shuffle=True)
```

**3. Initialize the trainer.** `Trainer` requires three positional arguments, `(data, saver, worker)`: the dataloader, a `Saver` for checkpoints and loss logs, and the sampler. `num_output` designates the trailing visible units used by the conditional (NCL) term; here it is `0`, so this run is pure unsupervised generation (see How It Works above).

```python
saver = Saver()
trainer = Trainer(dataloader, saver, sampler,
                  num_visible=128, num_hidden=32, num_output=0)
```

**4. Set the hyperparameters.** Learning rates go through `set_learning_parameters`; the KL / NCL balance goes through `set_cost_parameter`, where `alpha` weights the KL term against the NCL term. `beta` is accepted for API compatibility but is not used by the current objective.

```python
trainer.set_learning_parameters(learning_rate=0.02, weight_decay_rate=0.01, momentum_rate=0.0)
trainer.set_cost_parameter(alpha=0.5, beta=10)
```

**5. Train.** Training counts steps rather than epochs; `num_processes` parallelizes the positive-phase conditional sampling across workers.

```python
trainer.train(max_steps=101, save_path="./test", num_processes=10)
```

After training, `sample_bm.ipynb` loads the checkpoint, samples joint visible+hidden states with the same sampler, deduplicates them, and compares the visible slice with the training data via PCA.

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
- The training objective mixes KL divergence (generation) and NCL term (conditional reconstruction), controlled by `alpha`.
- The `num_output` parameter designates output units, enabling conditional generation (set to 0 for pure unsupervised generation).
- Multi-process sampling parallelizes the positive-phase conditional sampling; cosine warmup stabilizes early training.
