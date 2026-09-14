<!-- ```YAML
title: Quick Start
slug: kpp-getting-started-quickstart
sidebar_position: 3
hide: false
``` -->

# Quick Start

>This chapter helps you get started quickly with `Kaiwu-PyTorch-Plugin` through simple examples.

## Training Workflow & Concepts

### Training Flow Overview

```{mermaid}
graph TB
    Data["Training Data<br>x = binary tensor"]
    Pos["Positive Phase (PyTorch)<br>h = rbm.get_hidden(x)"]
    Neg["Negative Phase (SDK Backend)<br>s = rbm.sample(sampler)"]
    Obj["Objective & Update (PyTorch)<br>objective = rbm.objective(h, s)<br>objective.backward()<br>opt_rbm.step()"]
    
    Data --> Pos
    Pos --> Obj
    Neg --> Obj
    Obj -. "Gradient Flow & Weight Update<br>(Updates rbm.parameters())" .-> Pos
    
    subgraph Model_Internal ["Model Internal"]
        Neg
    end
    
    classDef dataFill fill:#e1f5fe,stroke:#01579b,stroke-width:1px;
    classDef posFill fill:#e3f2fd,stroke:#1e88e5,stroke-width:1px;
    classDef negFill fill:#fff3e0,stroke:#fb8c00,stroke-width:1px;
    classDef objFill fill:#e8f5e9,stroke:#43a047,stroke-width:1px;
    
    class Data dataFill;
    class Pos posFill;
    class Neg negFill;
    class Obj objFill;
```
### Three-Step Training Cycle

| Step                                | Description                                         | Code                                                                  |
| ----------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------- |
| **1. Positive Phase (PyTorch)**     | Compute hidden activations from data                | `h = rbm.get_hidden(x, bernoulli=True)`                               |
| **2. Negative Phase (SDK)**         | Sample from model distribution, offloaded to SA/CIM | `s = rbm.sample(sampler)`                                             |
| **3. Objective & Update (PyTorch)** | Compute objective, backpropagate, update parameters | `objective = rbm.objective(h, s)` → `backward()` → `optimizer.step()` |

### Code Entity Mapping

```{mermaid}
graph TB
    subgraph Model_Logic ["Model Logic"]
        direction LR
        B["RestrictedBoltzmannMachine"]
        D["BoltzmannMachine"]
    end
    subgraph Sampling_Interface ["Sampling Interface"]
        E["_to_ising_matrix()"]
        F["model.sample(sampler)"]
    end
    subgraph External_SDK ["External SDK"]
        G["kaiwu.cim.CIMOptimizer"]
        H["kaiwu.cim.PrecisionReducer"]
        I["kaiwu.classical.SimulatedAnnealingOptimizer"]
    end
    B -->|"implements"| E
    D -->|"implements"| E
    E -->|"prepares J/h for"| F
    F -->|"calls solve() on"| G
    F -->|"calls solve() on"| I
    G -->|"wrapped by"| H
```
**Mapping Walkthrough:**

1. **Model Logic:** BM/RBM holds `quadratic_coef` (W) and `linear_bias` (b,c)
2. **Ising Conversion:** `_to_ising_matrix()` converts W,b,c into Ising J (coupling) and h (field) matrix
3. **Sampling Interface:** `model.sample(sampler)` passes J,h to the sampler's `solve()` method
4. **SDK Backend:**
    - Classical: `SimulatedAnnealingOptimizer`
    - Quantum: `CIMOptimizer`
### BM vs RBM: Key API Difference

|                |               RBM               |             **BM**             |
| -------------- | :-----------------------------: | :----------------------------: |
| Topology       |            Bipartite            |        Fully connected         |
| Positive phase | `get_hidden(x, bernoulli=True)` | `condition_sample(sampler, x)` |
| Negative phase |        `sample(sampler)`        |       `sample(sampler)`        |
| Objective      |        `objective(h, s)`        |       `objective(h, s)`        |

> **Note:** BM uses `condition_sample()` because fully-connected networks require sampling conditioned on visible units, not a deterministic hidden-layer computation.

## Code Examples

### Restricted Boltzmann Machine (RBM)

The following example demonstrates how to use the `RestrictedBoltzmannMachine` class for basic model training. You can define the number of visible and hidden nodes, and customize the initialization of quadratic terms and linear biases.

```python
import torch

import kaiwu as kw
from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.cim import CIMOptimizer, PrecisionReducer
from torch.optim import SGD

# Add license authentication
# print("User ID:", os.getenv("USER_ID"), "SDK Code:", os.getenv("SDK_CODE"))
# kw.license.init(os.getenv("USER_ID"), os.getenv("SDK_CODE"))

if __name__ == "__main__":
    USE_QPU = False
    USE_CIM = False
    NUM_READS = 1
    SAMPLE_SIZE = 1

    if USE_CIM:
        kw.common.CheckpointManager.save_dir = './tmp'
        sampler = CIMOptimizer(task_name="test_kpp", wait=True)
        sampler = PrecisionReducer(
            sampler,
            precision=8,
            truncated_precision=10,
            target_bits=550,
            only_feasible_solution=False,
        )
    else:
        sampler = SimulatedAnnealingOptimizer(size_limit=NUM_READS)
    num_nodes = 5
    num_visible = 2
    x = 1.0 * torch.randint(0, 2, (SAMPLE_SIZE, num_visible))

    # Instantiate the model
    rbm = RestrictedBoltzmannMachine(
        num_visible,
        num_nodes - num_visible,
        quadratic_coef=torch.FloatTensor(
            [
                [2, -3, 0],
                [-1, 2, 0],
            ]
        ),
        linear_bias=torch.FloatTensor([1, 1, 0, -1, 2]),
    )
    # Instantiate the optimizer
    opt_rbm = SGD(rbm.parameters())

    # Example of one iteration in a training loop
    # Generate a sample set from the model
    x = rbm.get_hidden(x, bernoulli=True)
    s = rbm.sample(sampler)
    opt_rbm.zero_grad()
    # Compute the objective---this objective yields the same gradient as the negative
    # log likelihood of the model
    objective = rbm.objective(x, s)
    objective.backward()
    # Update model weights with a step of stochastic gradient descent
    opt_rbm.step()
```

### Boltzmann Machine (BM)

The following example demonstrates how to use the `BoltzmannMachine` class:

```python
import torch

from torch.optim import SGD
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import BoltzmannMachine

# Add license authentication here

if __name__ == "__main__":
    USE_QPU = False
    SAMPLE_SIZE = 5

    sampler = SimulatedAnnealingOptimizer(alpha=0.99, size_limit=5)
    sample_kwargs = {}
    num_nodes = 5
    num_visible = 2
    x = 1.0 * torch.randint(0, 2, (SAMPLE_SIZE, num_visible))

    # Instantiate the model
    bm = BoltzmannMachine(num_nodes)

    # Instantiate the optimizer
    opt_bm = SGD(bm.parameters())

    # Example of one iteration in a training loop
    # Generate a sample set from the model

    x = bm.condition_sample(sampler, x)
    s = bm.sample(sampler)
    opt_bm.zero_grad()
    # Compute the objective---this objective yields the same gradient as the negative
    # log likelihood of the model
    objective = bm.objective(x, s)
    # Backpropagate gradients
    print("call backward")
    objective.backward()
    print("after backward")
    # Update model weights with a step of stochastic gradient descent
    opt_bm.step()
    print(objective)
```

## Sampler Switching

The Kaiwu SDK provides various samplers. You can choose according to your needs:

```py
from kaiwu.classical import SimulatedAnnealingOptimizer

# Simulated-annealing optimizer (recommended for most scenarios)
sampler_sa = SimulatedAnnealingOptimizer()

# To use the quantum sampler (requires real-machine access)
# from kaiwu.cim import CIMOptimizer
```

## Next Steps

Congratulations on completing the quick start! For the next step:

- **Beginner Course:** Check out the [KPP Tutorials](tutorials/index.md) for more practical application examples. Start with `tests/test_rbm.py` for basic execution, then follow `example/rbm_digits/rbm_digits.ipynb` for a full RBM training walkthrough.
- **Intermediate Tutorial:** Explore generative modeling with fully connected Boltzmann Machines in `example/bm_generation/` (e.g., RNA sequence data augmentation), and learn feature learning & classification with RBM/DBN stacks.
- **Advanced Case Study:** Dive into the Q-VAE (Quantum Variational Autoencoder) training pipeline and Q-Diffusion for discrete sequence generation, where a Boltzmann Machine serves as a learnable prior or energy-guided sampler.

### Learning Paths

| Level            | Resources                                                             | Description                       |
| ---------------- | --------------------------------------------------------------------- | --------------------------------- |
| **Beginner**     | `tests/test_rbm.py`, basic execution                                  | Basic RBM execution               |
|                  | `example/rbm_digits/rbm_digits.ipynb`, full RBM walkthrough           | Complete RBM training      |
| **Intermediate** | `example/bm_generation/`, generative modeling with BMs (RNA sequence) | Generative modeling               |
|                  | RBM & DBN: feature learning & classification                          | Feature learning & classification |
| **Advanced**     | Q-VAE: Quantum Variational Autoencoder                                | Quantum VAE training              |
|                  | Q-Diffusion: discrete sequence generation                             | Discrete sequence generation      |

### Official Resources
- **KPP GitHub:** [https://github.com/QBoson/Kaiwu-pytorch-plugin](https://github.com/QBoson/Kaiwu-pytorch-plugin)
- **KPP Docs:** [https://kaiwu-pytorch-plugin-docs.readthedocs.io](https://kaiwu-pytorch-plugin-docs.readthedocs.io/)
- **SDK Docs:** [https://kaiwu-community.readthedocs.io](https://kaiwu-community.readthedocs.io/)