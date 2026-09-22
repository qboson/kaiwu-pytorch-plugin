<!-- ```YAML
title: Kaiwu-PyTorch-Plugin (KPP)
slug: kpp-getting-started-intro
sidebar_position: 1
hide: false
```
 -->

# Introduction

> **Kaiwu-PyTorch-Plugin (KPP)** is a PyTorch plugin designed for training and evaluating quantum-native energy models and enhanced AI models on **Special Purpose Quantum Computers (SPQC)**.
> 
> It offloads Boltzmann sampling to the **Coherent Ising Machine (CIM)** through the **Kaiwu SDK**, while keeping all other computations, parameter updates, autograd, data loading, in the standard **PyTorch** workflow.

## General Overview
### Design Goals

- **Integrate quantum sampling capabilities** while preserving the native PyTorch programming experience.
- **Support definition, training, and inference** of standard BM/RBM models
- **Extensible interfaces** for swapping samplers or energy functions
- **Decoupled architecture**: energy model definition is **separated** from sampling execution

### Features

```{list-table}
:widths: 20 80
:header-rows: 1
* - Feature
  - Description
* - **Model Support**
  - Standard RBM and fully connected BM
* - **Layer Configuration**
  - Customizable visible and hidden layer dimensions
* - **Sampling**
  - Negative-phase sampling executed on SPQC via Kaiwu SDK
* - **PyTorch Integration**
  - Model parameters are `torch.nn.Parameter`; supports autograd, GPU acceleration, and combination with other PyTorch modules
* - **Extended Models**
  - DBN, Q-VAE, Q-Diffusion
```
### Extension Mechanisms

- Decouple the energy function (e.g., Ising form) from the sampler.
- Allow users to replace sampling strategies by implementing standard interfaces (e.g., switching to classical MCMC or other backends).
- Support custom energy terms suitable for non-standard BM variants.

## System Architecture & Component Interaction

KPP follows a **decoupled architecture** where energy model definition is separated from sampling execution (hardware or simulation). Researchers can treat quantum samplers as **pluggable backends** within standard PyTorch workflows.

```{mermaid}
graph TD
    subgraph Classical_Framework ["Classical Framework"]
        PyTorch["PyTorch<br>(Autograd/Tensors)"]
    end
    subgraph Kaiwu_SDK ["Kaiwu SDK"]
        SA["SimulatedAnnealingOptimizer"]
        CIM["CIMOptimizer"]
    end
    subgraph Torch_Plugin ["Kaiwu Torch Plugin"]
        Abstract["abstract_boltzmann_machine.py"]
        RBM["restricted_boltzmann_machine.py<br>RestrictedBoltzmannMachine"]
        GBRBM["gbrbm.py<br>GaussianBernoulliRestricte"]
        DBN["dbn.py<br>UnsupervisedDBN"]
        FullBM["full_boltzmann_machine.py<br>BoltzmannMachine"]
        QDiff["qdiffusion.py<br>QDiffusion"]
        QVAE["qvae.py<br>QVAE"]
    end
    subgraph Examples
        rbm_digits["rbm_digits"]
        dbn_digits["dbn_digits"]
        bm_generation["bm_generation"]
        qdiffusion["qdiffusion"]
        qvae_mnist["qvae_mnist"]
        qvae_cell["qvae_cell"]
    end
    %% 顶部到插件
    PyTorch --> Abstract
    SA -.-> Abstract
    CIM -.-> Abstract
    %% 插件内部
    Abstract --> RBM
    Abstract --> GBRBM
    Abstract --> FullBM
    Abstract --> QDiff
    Abstract --> QVAE
    RBM --> DBN
    %% 插件到Examples
    RBM --> rbm_digits
    DBN --> dbn_digits
    FullBM --> bm_generation
    QDiff --> qdiffusion
    QVAE --> qvae_mnist
    QVAE --> qvae_cell
```

**Layer Breakdown:**
```{list-table}
:widths: 25 75
:header-rows: 1
* - Layer
  - Contents
* - **Application Layer**
  - `rbm_digits`, `dbn_digits`, `bm_generation`, `qvae_mnist`, `qvae_cell`, `qdiffusion`
* - **KPP Layer**
  - `AbstractBoltzmannMachine`, `RestrictedBoltzmannMachine`, `BoltzmannMachine`, `DBN`, `GBRBM`, `QVAE`, `QDiffusion`; `_to_ising_matrix()`, `sample()`, `objective()`
* - **Kaiwu SDK Layer**
  - Classical (SA) & Quantum (CIM) optimizers
```

### Core Concept 1: Quantum-Classical Hybrid Workflow

KPP follows the standard energy-based model training loop, with a quantum-accelerated negative phase.

#### Three-Step Training Cycle:

1. **Positive Phase (PyTorch, CPU/GPU):**
   - Compute hidden layer representations from training data.
   - `h = rbm.get_hidden(x)` → data statistics, no sampling needed.

2. **Negative Phase (Offloaded to SDK):**
   - Generate samples from the model distribution.
   - `s = rbm.sample(sampler)`.
   - Internally, `_to_ising_matrix()` prepares the J/h matrix and dispatches it to a **CIM** (quantum) or **SA** (classical) solver.

3. **Parameter Update (PyTorch, CPU/GPU):**
   - `objective = rbm.objective(h, s)`.
   - `objective.backward()`.
   - `optimizer.step()`.

```{mermaid}

graph TB

    %% 节点定义

    Data["Training Data<br>x = binary tensor"]

    Pos["Positive Phase (PyTorch)<br>h = rbm.get_hidden(x)"]

    Neg["Negative Phase (SDK Backend)<br>s = rbm.sample(sampler)"]

    Obj["Objective & Update (PyTorch)<br>objective = rbm.objective(h, s)<br>objective.backward()<br>opt_rbm.step()"]

    %% 连线

    Data --> Pos

    Pos --> Obj

    Neg --> Obj

    Obj -. "Gradient Flow & Weight Update<br>(Updates rbm.parameters())" .-> Pos

    %% 负相位来自模型（隐含），而非数据

    %% 可添加注释或虚框表示

    subgraph Model_Internal ["Model Internal"]

        Neg

    end

    %% 样式

    classDef dataFill fill:#e1f5fe,stroke:#01579b,stroke-width:1px;

    classDef posFill fill:#e3f2fd,stroke:#1e88e5,stroke-width:1px;

    classDef negFill fill:#fff3e0,stroke:#fb8c00,stroke-width:1px;

    classDef objFill fill:#e8f5e9,stroke:#43a047,stroke-width:1px;

    class Data dataFill;

    class Pos posFill;

    class Neg negFill;

    class Obj objFill;

```

> **Key Insight:** The only quantum part is `rbm.sample(sampler)`. Everything else is standard PyTorch.

### Core Concept 2: Energy Model Hierarchy

KPP provides a structured class hierarchy. All models share `sample(sampler)` and `objective(h, s)` interfaces.

```{mermaid}

classDiagram

    class AbstractBoltzmannMachine {
        <<abstract>>
        +forward(s_all)
        +sample(sampler)
        +objective(x, s)
        +_to_ising_matrix()
    }

    class RestrictedBoltzmannMachine {
        +num_visible: int
        +num_hidden: int
        +quadratic_coef: Parameter
        +linear_bias: Parameter
        +get_hidden(s_visible)
        +get_visible(s_hidden)
        +_to_ising_matrix()
    }

    class BoltzmannMachine {
        +num_nodes: int
        +condition_sample(sampler, s_visible)

    }

    AbstractBoltzmannMachine <|-- RestrictedBoltzmannMachine
    AbstractBoltzmannMachine <|-- BoltzmannMachine
```

**Model Descriptions:**

```{list-table}
:widths: 20 80
:header-rows: 1
* - Model
  - Description
* - **RBM**
  - Bipartite graph, layer-wise connections only
* - **BM**
  - Fully connected stochastic network; uses `condition_sample()`
* - **QVAE**
  - BM as prior in a VAE
* - **QDiffusion**
  - Energy-guided sampling for discrete sequence generation
```    

### Core Concept 3: Hardware Abstraction & Extension

- **Hardware abstraction:** KPP communicates with quantum hardware via Kaiwu SDK; swap samplers (SA/CIM) without code changes.
- **Sampler interface:**
```python
# Classical SA (development/testing)
from kaiwu.classical import SimulatedAnnealingOptimizer
sampler = SimulatedAnnealingOptimizer()

# Quantum CIM (real QPU)
from kaiwu.cim import CIMOptimizer, PrecisionReducer
sampler = CIMOptimizer(task_name="test_rbm", wait=True)
sampler = PrecisionReducer(
    sampler,
    precision=8,
    truncated_precision=10,
    target_bits=550,
    only_feasible_solution=False,
)
```
- **Extension mechanisms:** Decoupled energy function, swappable samplers, custom energy terms.

## Applications & Workflow
### Example Use Cases

- **RNA sequence data augmentation (BM):** Generative modeling with fully connected Boltzmann Machines for biological sequence augmentation.
- **Handwritten digit classification (RBM / DBN):** Train Restricted Boltzmann Machines and Deep Belief Networks on the MNIST dataset for feature learning and classification.
- **Q-VAE training pipeline:** Use a Boltzmann Machine as a learnable prior within a variational autoencoder.
- **Q-Diffusion:** Energy-guided sampling for discrete sequence generation.

### Target Users

```{list-table}
:widths: 30 70
:header-rows: 1
* - User Type
  - Use Case
* - **Researchers**
  - Validate the impact of quantum sampling on energy-based model training
* - **Developers**
  - Build hybrid classical-quantum generative models
* - **Educators & Students**
  - Teach Boltzmann machine principles and quantum sampling practice
```

### Typical Usage Workflow

The typical workflow for training energy-based models using the Kaiwu-PyTorch-Plugin is as follows:

1. <b>Data Preparation</b>: Load and preprocess the training data, converting it into the model input format
2. <b>Model Definition</b>: Instantiate an RBM or BM model, setting the dimensions of the visible and hidden layers.
3. <b>Optimizer Configuration</b>: Use PyTorch optimizers (such as SGD, Adam) to manage model parameters
4. <b>Training Loop</b>: 
    - Compute hidden layer representations from the training data (positive phase).
    - Use a sampler to generate samples from the model distribution (negative phase).
    - Calculate the loss function and backpropagate the gradients.
    - Update model parameters

5. <b>Model Evaluation</b>: Use the trained model to perform feature extraction, classification, or generation tasks

```python
import torch
from torch.optim import SGD
from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.classical import SimulatedAnnealingOptimizer

# Hyperparameters
batch_size = 32
num_visible = 784   # e.g., MNIST 28x28 flattened
num_hidden = 128
num_epochs = 10

# 1. Prepare data (binary visible units)
x = torch.randint(0, 2, (batch_size, num_visible)).float()

# 2. Define model
rbm = RestrictedBoltzmannMachine(num_visible, num_hidden)

# 3. Configure optimizer and sampler
optimizer = SGD(rbm.parameters(), lr=0.01)
sampler = SimulatedAnnealingOptimizer()

# 4. Training loop
for epoch in range(num_epochs):
    h = rbm.get_hidden(x, bernoulli=True)    # Positive phase: compute hidden layer
    s = rbm.sample(sampler)                  # Negative phase: model sampling

    optimizer.zero_grad()
    objective = rbm.objective(h, s)          # Yields gradient of negative log-likelihood
    objective.backward()                     # Backpropagation
    optimizer.step()                         # Update parameters
```

## Citation  

If Kaiwu-PyTorch-Plugin is helpful for your academic research, you are welcome to cite it: 

```{code-block} bibtex
 @software{KaiwuPyTorchPlugin,
     title = {Kaiwu-PyTorch-Plugin},
     author = {{QBoson Inc.}},
     year = {2024},
     url = {https://github.com/QBoson/Kaiwu-pytorch-plugin}
 }
```

Related research papers:
```{code-block} bibtex
 @article{QuantumBoostedDeepLearning,
     title = {Quantum-Boosted High-Fidelity Deep Learning},
     author = {{QBoson Research Team}},
     year = {2025},
     url = {https://arxiv.org/abs/2508.11190}
 }
```