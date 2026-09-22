# Why Quantum? Revisiting the Sampling Bottleneck

> By this point you have finished the [Getting Started](../index.md) guide and reviewed the [Prerequisites](../background.md). The [Theoretical Foundations](../../theoretical-foundations/index.md) established that Boltzmann machines are conceptually elegant but computationally crippled by the **sampling bottleneck**. This chapter recaps that conclusion, explains what quantum hardware changes, and shows how KPP lets you use classical and Coherent Ising Machine (CIM) samplers through the same training workflow. It is a bridge: the full derivations live in Theoretical Foundations, and the code lives in the next two chapters.

## The Sampling Problem in Exact Maximum Likelihood Learning

Recall from [Section 3.1, Defining the Objective: Low Energy for Real Data](../../theoretical-foundations/kpp-theoretical-foundations-ebms-def.md) that the gradient of the negative log-likelihood for an energy-based model decomposes into two expectations (Eq. {eq}`eq-nll-gradient-tf` in Theoretical Foundations):

```{math}
:label: eq-nll-gradient
\frac{\partial \mathcal{L}}{\partial \theta} =
\mathbb{E}_{\mathrm{data}}\!\left[ \frac{\partial E_\theta}{\partial \theta} \right] -
\mathbb{E}_{\mathrm{model}}\!\left[ \frac{\partial E_\theta}{\partial \theta} \right].
```

The data term is approximated by averaging over mini-batches. The model term requires sampling from the current Boltzmann distribution, and computing it exactly is #P-hard because it needs the partition function (Eq. {eq}`eq-partition`; [Section 3.2, The Intractable Partition Function Problem](../../theoretical-foundations/kpp-theoretical-foundations-ebms-partition.md)). We must therefore sample, and training quality hinges on how close the samples are to the target distribution, and how much they cost.

Classical approaches have known drawbacks (detailed in [Section 4.1](../../theoretical-foundations/kpp-theoretical-foundations-bm-overall.md), [Section 4.2](../../theoretical-foundations/kpp-theoretical-foundations-bm-restricted.md), and [Section 3.3](../../theoretical-foundations/kpp-theoretical-foundations-ebms-cd.md)):

- **MCMC / Gibbs sampling** converges to the target distribution, but mixing can be exponentially slow in rugged energy landscapes — *critical slowing down*.
- **CD / PCD** trade bias for speed: a few MCMC steps initialized at data points, risking **mode collapse**; PCD keeps persistent chains but still pays a sampling cost per update.

These methods are practical, but they should not be confused with exact sampling from the equilibrium distribution.

## The Quantum Alternative: Sampling by Physical Evolution

Instead of simulating a Markov chain, the energy function is encoded into a physical quantum system that evolves toward its low-energy states. Devices such as quantum annealers and Coherent Ising Machines (CIMs) realize this: model variables are mapped to qubits or optical pulses, and measurements yield samples from a distribution approximating the Boltzmann distribution at the hardware's effective temperature:

```{math}
:label: eq-effective-boltzmann
P(\mathbf{s}) \propto \exp\left(-\beta_{\mathrm{eff}} H(\mathbf{s})\right),
```

where $\beta_{\mathrm{eff}}$ is the effective inverse temperature ([Section 1.2, The Boltzmann Distribution and Equilibrium](../../theoretical-foundations/kpp-theoretical-foundations-stat-boltzmann.md), Eq. {eq}`eq-boltzmann-dist`).

The main promises are:

- **Direct physical sampling**: quantum tunneling may escape local minima more efficiently than thermal hopping.
- **Inherent parallelism**: thousands of optical pulses interact simultaneously, versus sequential Gibbs updates.
- **Hardware-scale sampling**: problem sizes that would be prohibitively slow to simulate classically.
- **Avoiding CD bias**: near-equilibrium samples without data-dependent initialization, giving less biased gradient estimates.

## From Boltzmann Machines to Ising Hamiltonians

The mapping follows directly from the spin-glass analogy ([Section 1.1, The Relationship Between Statistical Physics and Neural Networks](../../theoretical-foundations/kpp-theoretical-foundations-stat-spinglass.md); Eq. {eq}`eq-ising-hamiltonian`): binary variables $x_i \in \{0, 1\}$ become Ising spins $s_i = 2x_i - 1$, turning the quadratic energy into an Ising Hamiltonian:

```{math}
:label: eq-ising
H(\mathbf{s}) = -\sum_i h_i s_i - \sum_{i<j} J_{ij} s_i s_j + \mathrm{const},
```

the form that quantum annealing hardware and CIMs are designed to handle. KPP constructs this Ising matrix from the model parameters and hands it to the **Kaiwu SDK solver**; you never need to perform the conversion by hand.

## Classical and Quantum Samplers in KPP

KPP uses different sampling backends through the same `sample(sampler)` call, used in the [Quick Start](../quickstart.md) example. `SimulatedAnnealingOptimizer` suits local debugging and baseline experiments; `CIMOptimizer` requires valid Kaiwu SDK credentials and real-machine access. The two should be compared on the same model, sampling budget, and evaluation metrics, never inferred from the sampler type alone.

```python
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.cim import CIMOptimizer

classical_sampler = SimulatedAnnealingOptimizer()
quantum_sampler = CIMOptimizer(task_name="my_experiment", wait=True)
```

## Summary

- Sampling from the model distribution is the central computational bottleneck of Boltzmann machine training (Eq. {eq}`eq-nll-gradient`).
- Classical MCMC suffers from slow mixing times, and practical approximations such as CD introduce bias and can cause mode collapse.
- Quantum sampling encodes the energy as an Ising Hamiltonian and samples by physical evolution, promising better mixing, parallelism, and less bias.
- KPP converts model parameters into an Ising problem and connects classical and quantum backends through a unified sampling interface.
- The next chapter establishes the classical baseline, [Simulated Annealing for Ising Models](simulated_annealing.md), before the quantum integration in [Integrating Quantum Samplers into PyTorch](quantum_sampling_pytorch.md).
