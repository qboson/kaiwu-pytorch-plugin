<!-- ```YAML
title: Installation Guide
slug: kpp-getting-started-installation
sidebar_position: 2
hide: false
``` -->


# Installation Guide

> This chapter describes how to install Kaiwu-PyTorch-Plugin and its dependencies.

## System Requirements

Before installation, ensure your system meets the following requirements:

| Dependency | Required Version | Role |
|------------|-----------------|------|
| Python | 3.10 | Base runtime (tested against 3.10 specifically) |
| PyTorch | 2.7 | Tensor operations & automatic differentiation |
| NumPy | 2.2.6 | Linear algebra & SDK interfacing |
| Kaiwu SDK | v1.3.1+ | Ising optimization backend (classical SA + quantum CIM) |

> **Note:** Exact versions of PyTorch and NumPy are pinned in `requirements/requirements.txt`. Kaiwu SDK v1.3.1 is the current recommended release.

Check the Python version:
```bash
python --version
# or
python3 --version
```

If you need to install Python 3.10, please visit the <u>Python 3.10 download page</u>.

## Install Kaiwu-PyTorch-Plugin

> You can choose either local setup (conda/pip) or Docker setup (recommended for reproducibility and isolation).
### Option 1: Local Setup (conda/pip)

#### Step-by-Step Guide

1. **Create and Activate Environment**
It is recommended to use conda to create an isolated Python environment:
```bash
# Create a new environment
conda create -n quantum_env python=3.10

# Activate the environment
conda activate quantum_env
```

2. **Clone the Repository**
Clone the project from GitHub to your local machine:
```bash
git clone https://github.com/QBoson/Kaiwu-pytorch-plugin.git
cd kaiwu-pytorch-plugin
```

3. **Install Dependencies**
Install the project dependencies:
```bash
pip install -r requirements/requirements.txt
```

4. **Install the Plugin**
```bash
pip install .
```

#### Checkpoint

Run 
```bash
python -c "import torch; print(torch.__version__)"
```
to confirm PyTorch is installed.

### Option 2: Docker Setup (no local environment required)

The Docker setup builds a pre‑configured Jupyter notebook environment with all dependencies (including the Kaiwu SDK) already installed.

#### Required Project Structure

```bash
requirements/
├── docker-compose.yml
├── requirements.txt               # Kaiwu SDK included
├── kaiwu-1.3.1-py3-none-any.whl   # Or download from Qboson platform
└── docker/
    └── Dockerfile
```

#### Docker Commands

1. **Clone the repository**
Clone the project from GitHub to your local machine:
   ```bash
   git clone https://github.com/QBoson/Kaiwu-pytorch-plugin.git
   cd kaiwu-pytorch-plugin/requirements
   ```

2. **Build the Docker image**
   ```bash
   docker compose build
   ```

3. **Start the Jupyter notebook server**
   ```bash
   docker compose up
   ```

#### Access

- Open `http://localhost:8888` (no token required)
- Project root mounted at `/home/jovyan/work`
- Use JupyterLab Terminal for command-line work

```bash
/home/jovyan/work
# All project files are already here; edit, run, and version them directly.
```
This mounts your local repository into the container, so any changes you make locally will be reflected immediately inside Jupyter.

#### Stop the Server

Press `Ctrl+C`, then run:
```bash
docker compose down
```

## Project Structure & Package Layout

```bash
Kaiwu-pytorch-plugin/
├── src/kaiwu/torch_plugin/   # Core library
│   ├── __init__.py
│   ├── abstract_boltzmann_machine.py
│   ├── restricted_boltzmann_machine.py
│   ├── full_boltzmann_machine.py
│   ├── gbrbm.py
│   ├── dbn.py
│   ├── qvae.py
│   └── qdiffusion.py
├── example/                   # Application examples
│   ├── rbm_digits
│   ├── dbn_digits
│   ├── bm_generation/
│   ├── qvae_mnist/
│   ├── qvae_cell/
│   └── qdiffusion/
├── tests/                      # Test suite
│   └── test_rbm.py
├── requirements/               # Dependencies & Docker
├── docs/                       # Documentation
└── README.md
```

### Key Code Entities

| Entity                           | Description                                                                                                                        |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **`AbstractBoltzmannMachine`**   | Base class defining the interface for all models                                                                                   |
| **`BoltzmannMachine`**           | Fully connected; uses `condition_sample()` for positive phase                                                                      |
| **`RestrictedBoltzmannMachine`** | Bipartite graph with `quadratic_coef` and `linear_bias` for custom energy definition                                               |
| **`GaussianBernoulliRBM`**       | RBM variant with real-valued visible units (Gaussian distribution) and binary hidden units (Bernoulli distribution), or vice versa |

## Kaiwu SDK Configuration & License

Kaiwu-PyTorch-Plugin depends on the **Kaiwu SDK** to provide quantum computing capabilities. The Kaiwu SDK is a proprietary backend that provides both classical simulation (SA) and quantum execution (CIM), and must be installed separately.

### Step 1: Obtain & Install SDK
#### Option A: Install from PyPI (Recommended)

The Kaiwu SDK is available on PyPI. You can install it directly:

```bash
pip install kaiwu==1.3.1
```

Alternatively, add the following line to your `requirements.txt`:

```text
kaiwu==1.3.1
```

#### Option B: Install from Official Wheel

If you need a specific version or have network restrictions, download the SDK wheel from the QBoson platform:

> 1. Visit the <u>Kaiwu SDK download page</u> on the QBoson platform
> 2. Download the wheel package suitable for your system (e.g., `kaiwu-1.3.1-py3-none-any.whl`)
> 3. Install the wheel:

```bash
pip install kaiwu-1.3.1-py3-none-any.whl
```

### Step 2: Set Environment Variables

After installation, you need to configure the SDK authorization credentials. Set the following environment variables:

```bash
export USER_ID="<your-user-id>"
export SDK_CODE="<your-sdk-code>"
```

> **Important:** The environment variable name is `SDK_CODE`, **not** `SDK_TOKEN`. Using the wrong variable name will cause authentication to fail.

### Step 3: Initialize License in Code

Then initialize the license in your code:

```python
import os
import kaiwu as kw

kw.license.init(
    os.getenv("USER_ID"),
    os.getenv("SDK_CODE")
)
```

> **Note:** Always keep credentials in environment variables, never hardcode them in source code, as they could be accidentally committed to version control. The credentials can be obtained from the <u>Kaiwu SDK page</u> on the QBoson platform.

## Obtain Quantum Computer Access
### How to Get QPU Access
To experience true quantum computing capabilities, you need to obtain access to a quantum computer:

> 1. Register an account on the <u>QBoson platform</u>
> 2. Contact the official staff through the platform to request a quota for the real quantum device

> **Note:** Before obtaining access to the real quantum device, you can use simulators for development and testing. The Kaiwu SDK provides various classical optimizers (such as the simulated annealing optimizer) as classical alternatives to the quantum sampler.

### Development Workflow Recommendation:

```{mermaid}
graph LR
    A["Develop & Debug<br>(SA Simulator)"]
    B["Validate Model<br>(SA Simulator)"]
    C["Scale to QPU<br>(CIM Quantum)"]
    
    A --> B
    B --> C
    classDef dev fill:#e3f2fd,stroke:#1e88e5,stroke-width:1px;
    classDef val fill:#fff3e0,stroke:#fb8c00,stroke-width:1px;
    classDef qpu fill:#e8f5e9,stroke:#43a047,stroke-width:1px;
    
    class A dev;
    class B val;
    class C qpu;
```

> **Note:** Always validate your model with the classical simulator before consuming QPU quota.
## Verify Installation

After installation, run the following code to verify that the installation was successful:

### Three Verification Steps

1. **Check versions:** Confirm PyTorch (2.x) & Kaiwu SDK (v1.3.1+) are loaded
2. **Test SDK backend:** Verify the classical SA solver works
3. **Instantiate KPP model:** Confirm `RestrictedBoltzmannMachine` imports and initializes

### Smoke-Test Code

```python
# Verify PyTorch
import torch
print(f"PyTorch version: {torch.__version__}")

# Verify Kaiwu SDK
import kaiwu
import numpy as np
from kaiwu.classical import SimulatedAnnealingOptimizer
opt = SimulatedAnnealingOptimizer()
mat = np.array([[1, -1], [-1, 1]])
result = opt.solve(mat)
print(f"Kaiwu SDK version: {kaiwu.__version__}")
print(result)

# Verify Kaiwu-PyTorch-Plugin
from kaiwu.torch_plugin import RestrictedBoltzmannMachine
print("Kaiwu-PyTorch-Plugin imported successfully!")
# Simple test
rbm = RestrictedBoltzmannMachine(num_visible=10, num_hidden=5)
print(f"RBM created with {rbm.num_visible} visible and {rbm.num_hidden} hidden units")
```

### Expected Output

```bash
PyTorch version: 2.7.0
Kaiwu SDK version: 1.3.1
Solver result: [array([1, -1]), array([-1, 1])]
Kaiwu-PyTorch-Plugin imported successfully!
RBM created with 10 visible and 5 hidden units
```
If no error occurs, your environment is ready. You can now build models and validate with classical samplers, then switch to the quantum sampler.

## Development Environment Setup (Optional)

If you plan to participate in the development of the plugin, you can install the development dependencies:

```bash
pip install -r requirements/devel.txt
```

### Run tests:

```bash
# Run all tests
pytest tests/

# Run specific tests
pytest tests/test_rbm.py
```

### Lint code style:

```bash
pylint src/kaiwu/
```
