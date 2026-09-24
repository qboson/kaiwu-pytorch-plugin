<!-- # 理论基础

```{toctree}
:maxdepth: 2

kpp-theoretical-foundations-stat-spinglass
kpp-theoretical-foundations-stat-boltzmann
kpp-theoretical-foundations-stat-noise
kpp-theoretical-foundations-stat-renormalization
kpp-theoretical-foundations-nn-recap
kpp-theoretical-foundations-nn-recurrent
kpp-theoretical-foundations-nn-learning
kpp-theoretical-foundations-ebms-def
kpp-theoretical-foundations-ebms-partition
kpp-theoretical-foundations-ebms-cd
kpp-theoretical-foundations-bm-overall
kpp-theoretical-foundations-bm-restricted
kpp-theoretical-foundations-bm-deep
kpp-theoretical-foundations-recap
``` -->


# Theoretical Foundations

> This guide lays the theoretical groundwork for **KPP (Kaiwu-PyTorch-Plugin)**, covering statistical physics, neural networks, and energy‑based models.
>
> It traces the chain from spin‑glass models to Boltzmann machines, establishing the conceptual basis for the quantum‑enhanced techniques covered in the subsequent tutorials.

For a step‑by‑step introduction to KPP, see the [Getting Started](../getting_started/index.md) guide.

## Statistical Physics of Memory and Computation

```{toctree}
:maxdepth: 1
:hidden:

kpp-theoretical-foundations-stat-spinglass
kpp-theoretical-foundations-stat-boltzmann
kpp-theoretical-foundations-stat-noise
kpp-theoretical-foundations-stat-renormalization
```

- [Spin-Glass Analogy](kpp-theoretical-foundations-stat-spinglass.md): Mapping Ising models to neural states, bridging physics and computation.
- [Boltzmann Distribution and Equilibrium](kpp-theoretical-foundations-stat-boltzmann.md): Defining state probability via energy and the partition function.
- [The Need for Noise](kpp-theoretical-foundations-stat-noise.md): From deterministic Hopfield traps to stochastic search, how noise helps escape local minima.
- [Renormalization Group](kpp-theoretical-foundations-stat-renormalization.md): From microscopic spins to macroscopic features via coarse-graining, analogous to hierarchical feature learning.

## Neural Network Fundamentals

```{toctree}
:maxdepth: 1
:hidden:

kpp-theoretical-foundations-nn-recap
kpp-theoretical-foundations-nn-recurrent
kpp-theoretical-foundations-nn-learning
```

- [Linear Neurons Recap](kpp-theoretical-foundations-nn-recap.md): Perceptrons, the XOR problem, and the motivation for depth.
- [Recurrent Networks and Content-Addressable Memory](kpp-theoretical-foundations-nn-recurrent.md): Dynamics versus feed-forward computation.
- [Hebbian Learning as Sculpting Energy](kpp-theoretical-foundations-nn-learning.md): How local synaptic rules shape the global energy landscape.

## Energy-Based Models (EBMs)

```{toctree}
:maxdepth: 1
:hidden:

kpp-theoretical-foundations-ebms-def
kpp-theoretical-foundations-ebms-partition
kpp-theoretical-foundations-ebms-cd
```

- [Defining the Objective: Low Energy for Real Data](kpp-theoretical-foundations-ebms-def.md): Shaping the energy landscape to fit observations.
- [The Intractable Partition Function Problem](kpp-theoretical-foundations-ebms-partition.md): The computational barrier to exact inference.
- [Contrastive Divergence](kpp-theoretical-foundations-ebms-cd.md): A practical shortcut, approximating gradients without equilibrium.

## Boltzmann Machine Architecture: The Full Spectrum

```{toctree}
:maxdepth: 1
:hidden:

kpp-theoretical-foundations-bm-overall
kpp-theoretical-foundations-bm-restricted
kpp-theoretical-foundations-bm-deep
```

- [Classical Boltzmann Machine](kpp-theoretical-foundations-bm-overall.md): Symmetric connections between visible and hidden units, and the wake-sleep algorithm.
- [Restricted Boltzmann Machine (RBM)](kpp-theoretical-foundations-bm-restricted.md): The bipartite breakthrough enabling efficient layer-wise sampling.
- [Beyond Single Layers: Stacking for Deep Learning](kpp-theoretical-foundations-bm-deep.md): Building Deep Belief Networks (DBNs) and the connection to modern pre-training.

## Recap: From Statistical Physics to Sampling Bottlenecks

```{toctree}
:maxdepth: 1
:hidden:

kpp-theoretical-foundations-recap
```

[Read the Recap](kpp-theoretical-foundations-recap.md) for the key takeaways:

- Statistical physics provides the probabilistic lens (Boltzmann distribution) and the metaphor of the energy landscape.
- Neural network fundamentals provide the structural components (units, biases, weights) and the biological inspiration (Hebbian learning).
- Energy-based models frame the learning objective as minimizing energy for real data; the intractability of exact gradients motivates approximations like Contrastive Divergence, which provides a practical solution to the intractable partition function.
- Boltzmann Machine Architecture transitions from the computationally prohibitive full Boltzmann Machine to the Restricted Boltzmann Machine (RBM), which serves as the critical building block for stacking deep generative architectures.

These foundations expose a fundamental tension: **the expressiveness of energy-based models is bottlenecked by the difficulty of sampling from complex, high-dimensional Boltzmann distributions.** Classical Markov chain methods suffer from slow mixing and bias; quantum sampling offers a principled alternative by physically evolving a system toward equilibrium. 

---

## References and Further Reading

```{bibliography}
:style: unsrt
:list: bullet
:all:
```