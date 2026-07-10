<img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python Version"> <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License">

# Kaiwu-PyTorch-Plugin

**Language Versions**: [Chinese](README_ZH.md) | [English](README.md)

For using Kaiwu-PyTorch-Plugin, please refer to the [**documentation**](https://kaiwu-pytorch-plugin-docs.readthedocs.io/zh-cn/latest/).

## Project Overview

`Kaiwu-PyTorch-Plugin` is a quantum computing programming suite based on PyTorch and the Kaiwu SDK. It enables the training and evaluation of Restricted Boltzmann Machines (RBMs) and Boltzmann Machines (BMs) on coherent photonic quantum computers. The plugin provides easy-to-use interfaces, allowing researchers and developers to quickly implement the training and validation of energy-based neural network models and apply them to various machine learning development tasks.

A Restricted Boltzmann Machine is an energy-based unsupervised learning model consisting of a visible layer and a hidden layer, with full connections between layers but no connections within a layer. Its core idea is to model the probability distribution of data through an energy function and train weights using algorithms such as Contrastive Divergence (CD), allowing the model to learn hidden features of input data. RBMs are commonly used for feature extraction, dimensionality reduction, or collaborative filtering, and are also the foundation for building more complex models. A Boltzmann Machine is a fully connected stochastic neural network where all neurons can be interconnected (including within the visible and hidden layers). Traditional sampling methods for BMs are inefficient, and quantum computing provides a new approach.

```mermaid
flowchart TD
    torch[PyTorch tensors, modules, autograd]
    kaiwu[Kaiwu SDK samplers<br/>SA / CIM backend]

    subgraph plugin["src/kaiwu/torch_plugin"]
        abm["abstract_boltzmann_machine.py<br/>AbstractBoltzmannMachine"]
        bm["full_boltzmann_machine.py<br/>BoltzmannMachine"]
        rbm["restricted_boltzmann_machine.py<br/>RestrictedBoltzmannMachine"]
        qvae["qvae.py<br/>QVAE"]
        qdiff["qdiffusion.py<br/>QDiffusion"]
        dist["qvae_dist_util.py<br/>Bernoulli / mixture utilities"]
        dbn["dbn.py<br/>UnsupervisedDBN"]
    end

    torch --> abm
    kaiwu --> abm
    abm --> bm
    abm --> rbm
    abm --> qvae
    abm --> qdiff
    dist --> qvae
    rbm --> dbn

    subgraph examples["example"]
        rbm_digits["rbm_digits<br/>RBM feature learning and classification"]
        dbn_digits["dbn_digits<br/>stacked RBM pretraining and supervised DBN"]
        bm_generation["bm_generation<br/>BM distribution learning and sampling"]
        qdiffusion["qdiffusion<br/>protein discrete diffusion workflows"]
        qvae_mnist["qvae_mnist<br/>QVAE image generation and latent classification"]
        qvae_cell["qvae_cell<br/>single-cell QVAE representation learning"]
    end

    rbm --> rbm_digits
    dbn --> dbn_digits
    bm --> bm_generation
    qdiff --> qdiffusion
    qvae --> qvae_mnist
    qvae --> qvae_cell
    bm --> qvae_cell
```
The above image shows the project file structure:
- The Kaiwu-torch-plugin section of the code includes base class, Restricted Boltzmann Machine, and Boltzmann Machine.
- The example section of the code includes examples: qvae for generating digits, digits for digit recognition etc.
- The test section contains unit tests.

### Main Features
- Quantum Support: Inherits from Kaiwu SDK, supports calling photonic quantum computers
- Native PyTorch Support: Seamless integration with the PyTorch ecosystem, supports GPU acceleration
- Flexible Architecture: Supports custom visible and hidden layer dimensions
- Extensibility: Modular design makes it easy to add new energy functions or sampling methods
- Q-Diffusion Support: Includes a public `QDiffusion` module for energy-guided
  discrete generation with DPLM backbones

### Plugin Advantages

- Flexible Configuration: Sampling methods and energy functions are implemented separately, making it easy to add new energy functions or sampling methods. Widely used BMs and RBMs are implemented, and can be integrated into other models by defining objective functions.
- Example References: The plugin provides relevant examples, such as digits and qvae training, which can serve as references for your own work.
- Cutting-edge Algorithm Support: The plugin provides a solid platform for implementing and applying cutting-edge algorithms. For example, innovative methods that replace the Gaussian assumption in VAE with a Boltzmann distribution are implemented based on Kaiwu-Pytorch-Plugin. The plugin supports end-to-end model training for large-scale, high-noise single-cell data, lowering the barrier for algorithm development and application.

## Quick Start

### Requirements
- python == 3.10
- kaiwu == 1.3.1
- torch == 2.7.0
- numpy == 2.2.6

### Code Style

- Follows PEP 8 standards

### Installation Steps
You can choose either local setup (conda/pip) or Docker setup (recommended for reproducibility and isolation).

#### Option 1: Local Setup (conda/pip)
1. **Create and activate an environment**:
   ```bash
   # It is recommended to use conda to create a new environment
   conda create -n quantum_env python=3.10
   conda activate quantum_env
   ```

2. **Clone this repository locally**:
   ```bash
   git clone https://github.com/QBoson/Kaiwu-pytorch-plugin.git
   cd kaiwu-pytorch-plugin
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements/requirements.txt
   ```
   Kaiwu SDK needs to be installed separately, see the installation instructions below.

4. **Install the plugin**:
   ```bash
   pip install .
   ```

#### Option 2: Docker Setup (no local environment required)
The Docker setup builds a pre‑configured Jupyter notebook environment with all dependencies (including the Kaiwu SDK) already installed.

Project structure required:
```text
requirements/
├── docker-compose.yml
├── requirements.txt               # Kaiwu SDK included
├── kaiwu-1.3.1-py3-none-any.whl   # Or download from Qboson platform
└── docker/
    └── Dockerfile
```

1. **Clone the repository:**:
   ```bash
   git clone https://github.com/QBoson/Kaiwu-pytorch-plugin.git
   cd kaiwu-pytorch-plugin/requirements
   ```

2. **Enter the requirements directory and build the image**  
   ```bash
   cd kaiwu-pytorch-plugin/requirements
   docker compose build
   ```

3. **Start the Jupyter notebook server**:
   ```bash
   docker compose up
   ```
The notebook will be available at http://localhost:8888 (no token required).

4. **Directly access the plugin source code**  
   The container's `/home/jovyan/work` is mounted to the project root (`../`), so you can directly view and edit `src/`, `setup.py`, and all other files in the JupyterLab file browser — **no additional `git clone` inside the container is required**.

5. **Stop the server**:
Press `Ctrl+C`, then run:
   ```bash
   docker compose down
   ```

### Kaiwu SDK Installation Instructions (Required)

Kaiwu version 1.3.1 can now be installed directly via `pip install kaiwu==1.3.1`. 

The download and installation steps for other versions of the Kaiwu SDK are as follows:
![](imgs/image.png)

1. **Get the SDK**:
   - Visit [Kaiwu SDK download page](https://platform.qboson.com/sdkDownload) (registration required)
   - See [Kaiwu SDK installation instructions](https://kaiwu-sdk-docs.qboson.com/zh/latest/source/getting_started/sdk_installation_instructions.html)

2. **Configure authorization information**:
   Obtain your SDK authorization information:
   ```
   User ID: <your-user-id>
   SDK Token: <your-sdk-token>
   ```
   > Please replace the above information with your actual authorization details

### Obtain Real Quantum Machine Access

To experience real quantum computing, please register an account on the [Qboson Platform](https://platform.qboson.com/) and contact the official staff via the contact information provided in the documentation to obtain a real-machine quota.  

## Example Cases

### Simple Example
Below is a simple example of calling RBM. This example demonstrates how to use the interface and does not involve a specific task.

```python
import torch
from torch.optim import SGD
from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.cim import CIMOptimizer, PrecisionReducer

if __name__ == "__main__":
    SAMPLE_SIZE = 17
    USE_CIM = False

    if USE_CIM:
        sampler = CIMOptimizer(task_name="test_kpp", wait=True)
        sampler = PrecisionReducer(
            sampler,
            precision=8,
            truncated_precision=10,
            target_bits=550,
            only_feasible_solution=False
        )
    else:
        sampler = SimulatedAnnealingOptimizer()
    num_nodes = 50
    num_visible = 20
    x = 1 - 2.0 * torch.randint(0, 2, (SAMPLE_SIZE, num_visible))

    # Instantiate the model
    rbm = RestrictedBoltzmannMachine(
        num_visible,
        num_nodes - num_visible,
    )
    # Instantiate the optimizer
    opt_rbm = SGD(rbm.parameters())

    # Example of one iteration in a training loop
    # Generate a sample set from the model
    x = rbm.get_hidden(x)
    s = rbm.sample(sampler)

    opt_rbm.zero_grad()
    # Compute the objective---this objective yields the same gradient as the negative
    # log likelihood of the model
    objective = rbm.objective(x, s)
    # Backpropgate gradients
    objective.backward()
    # Update model weights with a step of stochastic gradient descent
    opt_rbm.step()
    print(objective)
```

### Q-Diffusion Quick Start

Q-Diffusion is available from the top-level plugin package as a generic
discrete-sequence core:

```python
from kaiwu.torch_plugin import QDiffusion, QDiffusionConfig
from kaiwu.torch_plugin.qdiffusion import SequenceTokenSpec

# Build your own proposal model, energy model, token spec, and energy adapter.
model = QDiffusion(
    proposal_model=proposal_model,
    energy_model=energy_model,
    token_spec=SequenceTokenSpec(
        pad_id=0,
        bos_id=1,
        eos_id=2,
        mask_id=3,
    ),
    energy_adapter=energy_adapter,
    config=QDiffusionConfig(num_candidates=4),
)
```

Runnable DPLM-based workflow examples live under `example/qdiffusion/`, with
`simple/` for minimal demos and `dplm/` for the protein-case workflows.

If you want to run those DPLM examples, install the extra example-side
dependencies separately:

```bash
pip install -r example/qdiffusion/requirements.txt
```

### Classification Task: Handwritten Digit Recognition  
Demonstrates feature learning and classification on the Digits dataset using Restricted Boltzmann Machines (RBM). This example is suitable for beginners to understand the application of RBMs in image feature extraction and classification, serving as a foundation for advanced experiments and functional extensions. Key steps include:  
- **Data Augmentation & Preprocessing**: Expand the original 8x8-pixel handwritten digit images through shifting (up, down, left, right) and normalize features using MinMaxScaler.  
- **RBM Model Training**: Implement the `RBMRunner` class to encapsulate the RBM training process, with support for visualizing generated samples and weight matrices during training.  
- **Feature Extraction & Classification**: After RBM training, use its hidden layer outputs as features for classification evaluation via logistic regression.  
- **Visual Analysis**: Enable sample generation and weight visualization during training to monitor and assess model learning.  

To run this example, execute `example/rbm_digits/rbm_digits.ipynb`.  

---  

### Generation Task: Q-VAE for MNIST Image Generation  

Demonstrates how to train and evaluate a Quantum Variational Autoencoder (Q-VAE) model on the MNIST handwritten digits dataset. This example is designed for users seeking to understand Q-VAE training, generation, and evaluation workflows, providing a foundation for generative model research. Key steps include:  
- **Data Loading & Preprocessing**: Implement a custom dataset class with batch indexing, combined with ToTensor conversion and flattening operations.  
- **Model Architecture**: Construct the Q-VAE framework, including encoder/decoder modules and RBM-based latent variable modeling.  
- **Training Process**: Design and execute the full training loop, tracking metrics (e.g., loss, ELBO, KL divergence) with support for checkpoint saving.  
- **Visualization & Generation**: Provide comparative visualization of original, reconstructed, and generated images for intuitive model assessment.  

To run this example, execute `example/qvae_mnist/train_qvae.ipynb`.  

![](imgs/qvae.png)

---  

### Q-Diffusion Generation Task: Proteomes: Homo sapiens Generation

Demonstrates how to train and evaluate an energy-guided discrete diffusion
workflow for protein sequence generation using `Q-Diffusion` with DPLM
backbones. This example is designed for users who want to understand how the
generic `Q-Diffusion` core is connected to practical protein-generation
experiments, providing a reference workflow for training, guided generation,
checkpoint reruns, and evaluation. Key steps include:

- **DPLM-backed Model Assembly**: Use
  `example/qdiffusion/dplm/utils/dplm_builder.py` to load one proposal backbone and
  one energy backbone, expose token metadata, build the energy adapter, and
  assemble a generic Q-Diffusion instance.
- **Training Objective**: In the epoch loop, tokenize FASTA sequences into
  `targets`, call `generator.objective({"targets": ...})`, corrupt clean
  sequences into noisy states, sample proposal candidates, and optimize
  `energy_objective.mean()` to train the energy-guidance branch.
- **Checkpoint and Rerun Workflow**: Save compact checkpoints containing the
  energy encoder, feature projector, energy backend weights, `energy_head`, and
  `vocab_proj`, then rebuild baseline and guided generators for test-time
  generation and reruns.
- **Evaluation and Reporting**: Compare baseline and guided outputs with quality
  metrics such as identity, Jensen-Shannon divergence, uniqueness, repeat
  ratio, and ESM2-based embedding distances, then write structured reports.

To run the minimal examples, execute:

```bash
pip install -r example/qdiffusion/requirements.txt
python example/qdiffusion/simple/simple_train_example.py
python example/qdiffusion/simple/simple_generate_example.py
```

To run the full DPLM workflow, execute:

```bash
python example/qdiffusion/dplm/train_workflow.py
```

For a more focused walkthrough of the example tree and its data flow, see
`example/qdiffusion/README.md`.

---

## Scientific Research Achievements  

### QBM Inside VAE = A More Powerful Generative Data Representer (QBM-VAE)  
Data from natural domains (e.g., biology, chemistry, materials science) exhibits extreme complexity, where traditional Gaussian i.i.d. assumptions often lead to distorted representations.  

By leveraging the native Boltzmann distribution sampler of coherent photonic quantum computers, we developed a Quantum Boltzmann Machine (QBM)-enhanced Deep Variational Autoencoder (QBM-VAE). This significantly improves the VAE’s encoding capability, enabling it to capture previously unrecognized deep data features.  

In single-cell transcriptomics analysis (a technique revealing cellular heterogeneity and functional differences by measuring gene expression at single-cell resolution), QBM-VAE markedly enhances clustering accuracy, identifying novel cell subtypes (new pathogenic factors with unique signatures) undetectable by conventional methods—providing new clues for target discovery.  

Based on this representation, we successfully integrated millions of single-cell transcriptomic data points and achieved superior performance in downstream tasks (e.g., cell clustering, classification, trajectory inference) compared to existing methods, validating the excellence of this latent representation.  

If you are interested in this work, please check out our paper:  
[**Quantum-Boosted High-Fidelity Deep Learning**](ttps://arxiv.org/pdf/2508.11190)

<img width="832" height="663" alt="1" src="https://github.com/user-attachments/assets/bc6097b3-6da8-4154-8aad-f749b4549fe1" />

---  

## Acknowledgments  
- We thank all contributors for their invaluable efforts.  
- We appreciate the support and feedback from the quantum computing community.  

## Contact  
1. **Boson Quantum Developer Community**: Access more learning resources.  
2. **Boson Quantum Official Assistant**: Inquire about real-machine access and collaborations.  
3. email: developer@boseq.com

 ![](imgs/qrcode.png) ![](imgs/qrcode3.png)  ![communication group](https://github.com/user-attachments/assets/bba37a66-777e-4b83-8535-e656119b0c78)

