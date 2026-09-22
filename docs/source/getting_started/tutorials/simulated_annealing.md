# Simulated Annealing for Ising Models

> [Why Quantum? Revisiting the Sampling Bottleneck](quantum_sampling_bottleneck.md) motivated the need for efficient sampling from Boltzmann distributions. This chapter introduces **simulated annealing (SA)**, the local CPU/GPU-only sampling baseline in KPP, which is also a natural reference point against which [quantum sampling](quantum_sampling_pytorch.md) is compared in the next chapter.

## The Ising Model at a Glance

Recall from [Section 1.1, The Relationship Between Statistical Physics and Neural Networks](../../theoretical-foundations/kpp-theoretical-foundations-stat-spinglass.md) that the Ising model describes interacting spins $s_i \in \{-1, +1\}$ with energy (Eq. {eq}`eq-ising-hamiltonian` in Theoretical Foundations):

```{math}
:label: eq-ising-energy
H(\mathbf{s}) = -\sum_i h_i s_i - \sum_{i<j} J_{ij} s_i s_j.
```

At temperature $T$, the probability of a spin configuration follows the Boltzmann distribution (see [Section 1.2, The Boltzmann Distribution and Equilibrium](../../theoretical-foundations/kpp-theoretical-foundations-stat-boltzmann.md), Eq. {eq}`eq-boltzmann-dist`):

```{math}
:label: eq-boltzmann-prob
P(\mathbf{s}) = \frac{1}{Z} \exp\left(-\frac{H(\mathbf{s})}{T}\right).
```

KPP converts Boltzmann machine parameters into this Ising matrix; the sampler solves it via the Kaiwu SDK `solve()` method.

## The Simulated Annealing Algorithm

Simulated annealing is a classical metaheuristic inspired by the annealing process in metallurgy. It proceeds in four steps:

1. **Initialization**: start at a high temperature $T_{\mathrm{high}}$ with a random spin configuration.
2. **Metropolis updates**: propose single-spin flips $s_i \to -s_i$; for an energy change $\Delta E$, accept with probability

```{math}
:label: eq-metropolis
P_{\mathrm{accept}} = \min\left(1, \exp\left(-\frac{\Delta E}{T}\right)\right).
```

3. **Cooling**: reduce the temperature by a schedule, e.g., geometric cooling $T_{k+1} = \alpha T_k$ with $0 < \alpha < 1$.
4. **Termination**: stop when the temperature is low; the final configuration approximates a low-energy state.

## From Optimization to Sampling

Optimization and Boltzmann sampling have different goals. If the goal is optimization, you keep cooling to find the ground state. If the goal is approximate sampling for generative modeling, two adaptations are needed:

- **Stop at a finite effective temperature** $T_{\mathrm{eff}}$ instead of cooling to zero.
- **Collect multiple samples** from independent annealing runs (or from a single run after thermalization at $T_{\mathrm{eff}}$).

When implemented carefully, the distribution of the collected samples approximates:

```{math}
:label: eq-sa-sampling
P(\mathbf{s}) \propto \exp\left(-\frac{H(\mathbf{s})}{T_{\mathrm{eff}}}\right),
```

which is precisely the Boltzmann distribution needed for training (Eq. {eq}`eq-boltzmann-prob`). The end-of-annealing configuration must not be treated automatically as an unbiased Boltzmann sample: the temperature, schedule, number of independent runs, and sample correlation all need to be considered.

## Using SA in KPP

The [Quick Start](../quickstart.md) example already trains an RBM with `SimulatedAnnealingOptimizer`. `sample()` constructs the Ising matrix and calls the optimizer; you only pass the sampler to the model:

```python
import torch
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import RestrictedBoltzmannMachine

rbm = RestrictedBoltzmannMachine(num_visible=20, num_hidden=30)
sampler = SimulatedAnnealingOptimizer()
samples = rbm.sample(sampler)
```

Here, `samples` is the state tensor returned by the model; KPP calls `sampler.solve(ising_mat)` internally, rather than `sampler.sample(hamiltonian)`.

## Relationship to Quantum Sampling

SA is a suitable starting point for local development, debugging, and baseline experiments. Real quantum hardware, i.e., the CIM, requires credentials, task queueing, and quota; whether it provides a benefit must be evaluated on the specific model, sample-quality metrics, and end-to-end latency. See the side-by-side comparison in [Integrating Quantum Samplers into PyTorch](quantum_sampling_pytorch.md) in the next chapter.

## Summary

- SA explores the Ising energy landscape via Metropolis updates and a cooling schedule.
- Optimization and Boltzmann sampling have different goals; finite-temperature settings and sample evaluation are essential.
- SA is used through `rbm.sample(sampler)` like any other optimizer.
- The trade-offs between SA and quantum sampling: thermal activation vs. tunneling, local vs. cloud, latency, are compared in the [next chapter](quantum_sampling_pytorch.md).