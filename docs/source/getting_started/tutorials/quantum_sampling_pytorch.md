# Integrating Quantum Samplers into PyTorch

> The previous two chapters motivated quantum sampling ([Why Quantum?](quantum_sampling_bottleneck.md)) and introduced SA as the classical baseline ([Simulated Annealing for Ising Models](simulated_annealing.md)). This chapter bridges physics and code: it introduces the **Kaiwu-PyTorch-Plugin (KPP)**, the software layer that plugs samplers, classical and quantum, into the PyTorch deep learning ecosystem. The [Quick Start](../quickstart.md) showed the basic training loop; here we explain the core interfaces, how to create and configure a CIM sampler, and what happens behind the scenes when you replace the local SA sampler with a quantum one.

## Core Interfaces

`RestrictedBoltzmannMachine` and `BoltzmannMachine` inherit from `AbstractBoltzmannMachine` (a `torch.nn.Module`). The RBM energy follows from [Section 4.2, Restricted Boltzmann Machine](../../theoretical-foundations/kpp-theoretical-foundations-bm-restricted.md) (Eq. {eq}`eq-rbm-energy-tf`):

```{math}
:label: eq-rbm-energy
E(\mathbf{v}, \mathbf{h}) = -\mathbf{b}^\top \mathbf{v} - \mathbf{c}^\top \mathbf{h} - \mathbf{v}^\top \mathbf{W} \mathbf{h}.
```

The interfaces most used in training are:

- `get_hidden(s_visible, ...)`: builds the positive-phase state containing both visible and hidden states for an RBM.
- `sample(sampler)`: constructs the Ising matrix from the current model parameters, calls `sampler.solve()`, and returns the negative-phase state tensor.
- `objective(s_positive, s_negative)`: computes a surrogate objective whose gradient is equivalent to the negative log-likelihood gradient.

Note that `sample()` does **not** take a data state as input, nor does it return separate visible and hidden tensors; `objective()` takes exactly the two phase tensors.

## Creating a CIM Sampler

Real hardware quantum sampling requires configured Kaiwu SDK credentials and real-machine access. `PrecisionReducer` adapts the Ising parameters to hardware precision constraints:

```python
from kaiwu.cim import CIMOptimizer, PrecisionReducer

quantum_sampler = CIMOptimizer(task_name="rbm_mnist", wait=True)
sampler = PrecisionReducer(
    quantum_sampler,
    precision=8,
    truncated_precision=10,
    target_bits=550,
    only_feasible_solution=False,
)
```

## A Minimal Training Iteration

The loop mirrors the [Quick Start](../quickstart.md) example: positive phase from data, negative phase from `sample(sampler)`. Swap in `SimulatedAnnealingOptimizer()` to run the same loop locally.

```python
import torch
from torch.optim import SGD
from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.classical import SimulatedAnnealingOptimizer

num_visible, num_hidden = 20, 30
rbm = RestrictedBoltzmannMachine(num_visible, num_hidden)
optimizer = SGD(rbm.parameters(), lr=0.01)
sampler = SimulatedAnnealingOptimizer()

v_data = torch.randint(0, 2, (16, num_visible)).float()
s_positive = rbm.get_hidden(v_data, bernoulli=True)
s_negative = rbm.sample(sampler)

optimizer.zero_grad()
loss = rbm.objective(s_positive, s_negative)
loss.backward()
optimizer.step()
```

## What Happens During `rbm.sample(sampler)`

From the PyTorch perspective, `sample()` behaves like any other tensor-producing function.

1. **Hamiltonian Construction**: weights and biases are converted into the Ising parameters $\{h_i, J_{ij}\}$ via the mapping described in the previous chapters.
2. **Precision Scaling**: `PrecisionReducer` discretizes the parameters to the hardware's bit precision.
3. **Cloud Submission**: the Hamiltonian is sent via the Kaiwu SDK; the job queues for the next available CIM unit.
4. **Physical Sampling**: the CIM evolves optically, producing spin configurations drawn from the approximate Boltzmann distribution at the hardware's effective temperature.
5. **Result Retrieval**: measured configurations return as PyTorch tensors.

## Choosing a Sampler

The table below consolidates the SA-vs-CIM comparison introduced in the [previous chapter](simulated_annealing.md):

| Consideration            | `SimulatedAnnealingOptimizer`                                       | `CIMOptimizer`                                        |
| ------------------------ | ------------------------------------------------------------------- | ----------------------------------------------------- |
| Run location             | Local CPU                                                           | Kaiwu real-machine service (cloud processor)          |
| Prerequisite             | None                                                                | SDK credentials, access rights, quota                 |
| Escape from local minima | Thermal activation (barrier crossing $\propto \exp(-\Delta E / T)$) | Quantum tunneling                                     |
| Suitable phase           | Debugging, baselines, small experiments                             | Comparative experiments with real-machine access      |
| End-to-end latency       | Determined by local problem size and settings                       | High — job submission, queuing, and physical evolution |

A common workflow is to **develop and debug locally with SA**, verifying data handling, the model, and the training loop, then **switch to the CIM sampler in the same experimental setup** by changing only the sampler instantiation. This makes it far easier to isolate problems.

## Summary

- KPP plugs samplers into the PyTorch training loop through `sample(sampler)`.
- The negative phase comes from `sample()`, and the training objective is computed by `objective(s_positive, s_negative)`.
- Quantum sampling follows a five-step pipeline: construction, scaling, submission, evolution, retrieval. All handled automatically.
- CIM samples should always be compared against a classical baseline on the same task and metrics.
