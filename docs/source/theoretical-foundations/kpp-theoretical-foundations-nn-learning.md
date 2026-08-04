---
title: 2.3 Hebbian Learning as Sculpting Energy
slug: kpp-theoretical-foundations-nn-learning
sidebar_position: 6
hide: false
hide_child: false
---


# 2.3 Hebbian Learning as Sculpting Energy

> The previous section introduced the Hopfield network as a dynamical system whose state evolves to minimize an energy function. The weights of the network determine the shape of this energy landscape, where the valleys are located, how deep they are, and how steep the walls between them. The question then arises: **how should these weights be set so that the energy landscape encodes useful memories?**
> The answer lies in a remarkably simple and biologically plausible principle proposed by Canadian psychologist Donald Hebb in 1949. The **Hebbian learning rule**, often summarized as *"neurons that fire together, wire together,"* provides a local mechanism by which neural activity sculpts the global energy function. This section explores the mathematical connection between Hebbian synaptic modification and the shaping of an energy landscape, establishing the foundational learning principle that carries forward into the Boltzmann machine.

## The Hebbian Postulate
In his seminal work "*The Organization of Behavior"*, Hebb proposed that:

> "When an axon of cell A is near enough to excite cell B and repeatedly or persistently takes part in firing it, some growth process or metabolic change takes place in one or both cells such that A's efficiency, as one of the cells firing B, is increased."

Translated into the language of artificial neural networks, this becomes a simple weight update rule: if two connected neurons are simultaneously active, the strength of their connection should increase. Conversely, if their activities are anti-correlated, the connection strength should decrease.

For binary neurons taking values in $\{-1, +1\}$, the change in weight $w_{ij}$ due to a single pattern $\boldsymbol{\xi}$ is proportional to the product of their activities:

$$\Delta w_{ij} \propto \xi_i \xi_j$$

For $\{0, 1\}$ valued neurons, a common variant adjusts weights only when both neurons are active:

$$\Delta w_{ij} \propto x_i x_j$$

In both formulations, the essential principle is the same: **co-activation strengthens the connection**.

## Storing Multiple Patterns: The Outer Product Rule
To store a set of $P$ patterns $\{\boldsymbol{\xi}^1, \boldsymbol{\xi}^2, \ldots, \boldsymbol{\xi}^P\}$, the Hebbian prescription is simply to sum the outer products of each pattern:

$$w_{ij} = \frac{1}{N} \sum_{\mu=1}^P \xi_i^\mu \xi_j^\mu$$

The factor $1/N$ (or sometimes $1/P$ is a normalization constant that prevents weights from growing without bound as more patterns are added. The weight matrix can be written compactly as:

$$\mathbf{W} = \frac{1}{N} \sum_{\mu=1}^P \boldsymbol{\xi}^\mu (\boldsymbol{\xi}^\mu)^\top - \frac{P}{N} \mathbf{I}$$

where the subtraction of the identity matrix enforces the condition $w_{ii} = 0$ (no self-connections).

This learning rule is purely **local**: to update $w_{ij}$, only the activities of neurons $i$ and $j$ are required. No global error signal, no backpropagation of gradients, no explicit knowledge of the energy function is necessary. Yet, as we shall see, this local rule has a profound global effect on the energy landscape.

## Hebbian Learning as Energy Basin Creation
Consider the energy function of the Hopfield network with Hebbian weights. Substituting the outer product weight formula into the energy definition yields:

$$\begin{aligned}E(\mathbf{x}) &= -\frac{1}{2} \sum_{i \neq j} w_{ij} x_i x_j - \sum_i b_i x_i \\&= -\frac{1}{2N} \sum_{\mu=1}^P \sum_{i \neq j} \xi_i^\mu \xi_j^\mu x_i x_j - \sum_i b_i x_i\end{aligned}$$

For a state $\mathbf{x}$ that exactly matches one of the stored patterns, say $\boldsymbol{\xi}^\nu$, the double sum simplifies dramatically. Assuming the patterns are random and uncorrelated, the dominant contribution comes from the term $\mu = \nu$:

$$\sum_{i \neq j} \xi_i^\nu \xi_j^\nu \xi_i^\nu \xi_j^\nu = \sum_{i \neq j} (\xi_i^\nu)^2 (\xi_j^\nu)^2 = N(N-1) \approx N^2$$

Thus, the energy of a stored pattern is approximately:

$$E(\boldsymbol{\xi}^\nu) \approx -\frac{1}{2N} \cdot N^2 = -\frac{N}{2}$$

This is a **deep energy minimum**. In contrast, a random state uncorrelated with any stored pattern will have energy near zero, because the cross terms average out.

The Hebbian rule therefore accomplishes the following geometric transformation:

- **Basins of attraction** are carved into the energy landscape at the locations of the stored patterns.
- The **depth** of each basin is proportional to $N$, the number of neurons.
- The **walls** between basins are shaped by the crosstalk terms from other patterns.

In this precise sense, **Hebbian learning sculpts the energy landscape**. Each memory contributes a quadratic depression centered at the pattern vector. The collective effect of all stored memories is an additive superposition of these depressions, resulting in a rugged landscape whose local minima correspond, ideally, to the stored memories.

## The Mathematical Foundation for Hebb's Rule
While Hebb's postulate is biologically motivated, the Boltzmann machine provides a rigorous mathematical justification for it. Recall from Section [3.1 Defining the Objective: Low Energy for Real Data](kpp-theoretical-foundations-ebms-def.md) the gradient of the log-likelihood for a Boltzmann machine:

$$\frac{\partial f}{\partial w_{ij}} = \langle x_i x_j \rangle_{\text{data}} - \langle x_i x_j \rangle_{\text{model}}$$

This learning rule, often called the **contrastive Hebbian rule**, is a direct generalization of Hebb's original idea. The first term, $\langle x_i x_j \rangle_{\text{data}}$, is purely **Hebbian**: it strengthens the connection between two units when they are co-active in the data. The second term, $\langle x_i x_j \rangle_{\text{model}}$, is **anti-Hebbian**: it weakens connections that are spuriously co-active in the model's own generated fantasies.

As derived in Section [3.3 Contrastive Divergence](kpp-theoretical-foundations-ebms-cd.md), the stochastic gradient update for a Boltzmann machine can be written as:

$$\Delta w_{ij} = \eta \left( X_i(\omega) X_j(\omega) - \mathbb{E}_\theta [X_i X_j] \right)$$

This formulation reveals that **the simple, local Hebbian update ("fire together, wire together") emerges naturally from the principled objective of maximizing the data likelihood**. In essence, the Boltzmann machine's learning algorithm proves that Hebb's rule is not just a biological heuristic but the optimal learning strategy when the goal is to match the model's equilibrium distribution to the data distribution.

## Limitations of Hebbian Learning
While elegant and biologically motivated, pure Hebbian learning has significant drawbacks when used in Hopfield networks:

1. **Spurious Minima**: The crosstalk between stored patterns creates unintended energy minima, mixture states and random valleys, that trap dynamics.
2. **Storage Capacity**: The number of patterns that can be reliably stored scales only linearly with $N$, with a proportionality constant of approximately 0.14. Beyond this, the network undergoes a phase transition to a spin-glass state where no patterns are retrievable.
3. **Catastrophic Forgetting**: Storing a new pattern modifies all weights and can destabilize previously stored memories. There is no mechanism to protect old memories while learning new ones.
4. **No Unlearning of Spurious States**: Once a spurious minimum is created, pure Hebbian learning offers no way to eliminate it. The rule only adds basins; it never fills them in.These limitations motivated the development of more sophisticated learning algorithms, including the **Boltzmann machine learning rule**, which uses a combination of Hebbian and anti-Hebbian updates guided by the difference between data-driven and model-driven correlations.

## From Hebbian Storage to Probabilistic Learning
The Boltzmann machine inherits the energy-based architecture of the Hopfield network and the idea that learning corresponds to modifying the energy landscape. However, it replaces the simple Hebbian storage prescription with a **principled probabilistic learning algorithm**.

In a Boltzmann machine, the goal is not merely to store a set of patterns as attractors, but to make the equilibrium distribution of the network match the distribution of the observed data. The learning rule that achieves this is a generalization of Hebb's principle:

$$\Delta w_{ij} \propto \langle x_i x_j \rangle_{\text{data}} - \langle x_i x_j \rangle_{\text{model}}$$

This is sometimes called the **contrastive Hebbian rule** or **delta rule**. The first term, $\langle x_i x_j \rangle_{\text{data}}$, is the average correlation between units $i$ and $j$when the visible units are clamped to data examples. This term is **Hebbian**: it strengthens connections between units that are co-active in the data.

The second term, $ \langle x_i x_j \rangle_{\text{model}}$, is the average correlation when the network runs freely from its own equilibrium distribution. This term is **anti-Hebbian**: it weakens connections that are spuriously co-active in the model's own fantasies.

The difference between these two correlations drives the weight update. If the data shows a stronger correlation than the model's current distribution, the weight increases (lowering energy for that co-activation pattern). If the model overestimates the correlation, the weight decreases (raising energy for that pattern).

This contrastive update dynamically sculpts the energy landscape to match the data distribution. It retains the local, Hebbian character of the update—only the activities of the connected neurons are required—while providing a principled way to avoid the pitfalls of pure Hebbian storage.

## Sculpting Energy: A Unifying Metaphor
The metaphor of **sculpting an energy landscape** provides a powerful unifying framework for understanding both Hopfield networks and Boltzmann machines:

- **Weights as sculpting tools**: Each weight $w_{ij}$ acts like a chisel that can either carve a valley (if $w_{ij} > 0$) or raise a ridge (if $w_{ij} < 0$) along the diagonal where $x_i = x_j$.
- **Hebbian learning as an additive process**: Each stored pattern adds a new depression to the landscape, creating a new attractor basin.
- **Boltzmann learning as a corrective process**: By comparing data correlations to model correlations, the learning algorithm not only adds basins where data lies but also **fills in spurious basins** that the model has invented.

This energy-centric view will recur throughout the remainder of this tutorial. The restricted Boltzmann machine (RBM), deep belief networks, and even modern energy-based models all share this **fundamental perspective**: learning is the process of shaping an energy function so that desirable configurations are assigned low energy and undesirable configurations are assigned high energy.

## Summary
- **Hebbian learning** is a local, biologically plausible rule: *neurons that fire together, wire together*.
- For binary patterns, the Hebbian weight update is $ \Delta w_{ij} \propto x_i x_j$.
- Storing multiple patterns corresponds to summing their outer products, resulting in a weight matrix that reflects pairwise correlations.
- **Hebbian learning sculpts the energy landscape** by creating basins of attraction at the locations of stored patterns.
- A single weight increase $\Delta w_{ij} > 0$ lowers the energy of states where $x_i$ and $x_j$ agree and raises the energy where they disagree.
- Pure Hebbian learning suffers from **spurious minima**, **limited capacity**, and **catastrophic forgetting**. 
- The **Boltzmann machine learning rule** generalizes Hebb's principle by subtracting model correlations from data correlations, providing a principled way to shape the energy landscape to match a probability distribution.
- The metaphor of **energy sculpting** unifies the learning principles of Hopfield networks and Boltzmann machines, and extends to modern deep generative models.
