---
title: '4.1 The Classic Boltzmann Machine: Visible and Hidden Symmetry'
slug: kpp-theoretical-foundations-bm-overall
sidebar_position: 10
hide: false
hide_child: false
---


# 4.1 The Classic Boltzmann Machine: Visible and Hidden Symmetry

> The preceding chapters have built the theoretical foundations—statistical physics, recurrent neural dynamics, and energy-based learning, that culminate in the **Boltzmann machine**. In this section, we present the architecture of the classic, fully connected Boltzmann machine as originally proposed by Geoffrey Hinton and Terrence Sejnowski in 1985. We will examine its structure, its probabilistic semantics, the learning algorithm that shapes its energy landscape, and the practical limitations that motivated the development of restricted variants.

## The Architecture: A Fully Connected Stochastic Network
A classic Boltzmann machine is a recurrent neural network composed of **stochastic binary units**. Each unit $x_i$ can be in one of two states: active (1) or inactive (0). The units are partitioned into two sets:

- **Visible units** $\mathbf{v} \in \{0,1\}^{N_v}$: These interface with the external world. They receive data during training and represent the observed variables of interest.
- **Hidden units** $\mathbf{h} \in \{0,1\}^{N_h}$: These are internal, latent variables that capture higher-order structure and dependencies not directly observed in the visible data.

Crucially, the classic Boltzmann machine is **fully connected**: every unit is symmetrically connected to every other unit, regardless of whether they are visible or hidden. The weight matrix $\mathbf{W}$ is a square, symmetric matrix of size $N \times N$, where $N = N_v + N_h$. The symmetry condition,

$$w_{ij} = w_{ji}, \quad \text{for all } i, j$$

is essential because it guarantees the existence of a well-defined energy function and ensures that the stochastic dynamics converge to an equilibrium distribution. Self-connections are prohibited (\( w_{ii} = 0 \)), as they would merely add a constant offset to the energy without affecting the conditional distributions that drive sampling.

Each unit also has a **bias** $b_i$, which can be interpreted as a connection from an always-active unit or as an external field acting on that unit.

## The Energy Function
The energy of a joint configuration $(\mathbf{v}, \mathbf{h})$ is defined as:

$$E(\mathbf{v}, \mathbf{h}) = -\sum_{i \in \text{vis}} b_i v_i - \sum_{j \in \text{hid}} c_j h_j - \sum_{i<j} w_{ij} x_i x_j$$

where $\mathbf{x} = (\mathbf{v}, \mathbf{h})$ denotes the full state vector, $b_i$ are visible biases, and $c_j$ are hidden biases. The summation over $i<j$ runs over all distinct pairs of units, both visible-visible, visible-hidden, and hidden-hidden.

This energy function defines a joint probability distribution over all $2^{N}$ possible states via the Boltzmann distribution:

$$P(\mathbf{v}, \mathbf{h}) = \frac{1}{Z} \exp\left(-E(\mathbf{v}, \mathbf{h})\right)$$

where the partition function $Z$ sums over all joint configurations:

$$Z = \sum_{\tilde{\mathbf{v}}, \tilde{\mathbf{h}}} \exp\left(-E(\tilde{\mathbf{v}}, \tilde{\mathbf{h}})\right)$$

The probability of observing a visible vector $\mathbf{v}$ alone is obtained by marginalizing over the hidden units:

$$P(\mathbf{v}) = \sum_{\tilde{\mathbf{h}}} P(\mathbf{v}, \tilde{\mathbf{h}}) = \frac{1}{Z} \sum_{\tilde{\mathbf{h}}} \exp\left(-E(\mathbf{v}, \tilde{\mathbf{h}})\right)$$

## Stochastic Dynamics and Equilibrium
The network operates as a **stochastic dynamical system**. At each time step, a unit is selected at random, and its new state is sampled from its conditional distribution given the states of all other units. For a unit $i$, the probability of being active is:

$$P(x_i = 1 \mid \mathbf{x}_{-i}) = \sigma\left( \sum_{j \neq i} w_{ij} x_j + b_i \right)$$

where $\sigma(z) = 1/(1 + \exp(-z))$ is sigmoid function, and $\mathbf{x}_{-i}$ denotes the states of all units except $i$.

Because the weights are symmetric, this update satisfies detailed balance with respect to the Boltzmann distribution defined in Section [1.2 The Boltzmann Distribution and Equilibrium](kpp-theoretical-foundations-stat-boltzmann.md). 

Consequently, if the network is allowed to run for a sufficiently long time (i.e., many asynchronous updates), the sequence of states constitutes a Markov chain whose stationary distribution is exactly $P(\mathbf{v}, \mathbf{h})$. This property is the cornerstone of both inference and learning in the Boltzmann machine.

Consequently, repeated application drives the network toward equilibrium, where configurations are sampled according to $P(\mathbf{v}, \mathbf{h})$.

## Learning Objective: Maximum Likelihood with Hidden Variables
The goal of learning is to adjust the weights and biases so that the marginal distribution over visible units, $P(\mathbf{v})$, approximates the empirical distribution of the training data. As derived in Section [3.1 Defining the Objective: Low Energy for Real Data](kpp-theoretical-foundations-ebms-def.md), the gradient of the negative log-likelihood for a single training example $\mathbf{v}$ with respect to a weight $w_{ij}$ is:

$$\frac{\partial \left( -\log P(\mathbf{v}) \right)}{\partial w_{ij}} = \mathbb{E}_{\mathbf{h} \mid \mathbf{v}} \left[ x_i x_j \right] - \mathbb{E}_{P(\mathbf{v}, \mathbf{h})} \left[ x_i x_j \right]$$

Here, $ x_i$ and $x_j$ denote the states of the two connected units. The first term is the expected product of their activities when the visible units are **clamped** to the training example $\mathbf{v}$ and the hidden units are allowed to fluctuate according to their conditional distribution. This is the **positive phase** or **wake phase** statistic.

The second term is the expected product when the network runs **freely** without any external input, sampling from its equilibrium distribution. This is the **negative phase** or **sleep phase** statistic.

The learning rule is thus:

$$\Delta w_{ij} = \eta \left( \langle x_i x_j \rangle_{\text{clamped}} - \langle x_i x_j \rangle_{\text{free}} \right)$$

where $\eta$ is the learning rate. Biases are updated similarly by treating them as weights from an always-on unit with state 1.

## The Wake-Sleep Algorithm: A Biological Metaphor
The two-phase nature of the learning rule suggests a compelling biological metaphor. In the **wake phase**, the network is driven by sensory input (the visible units are clamped to data). The hidden units respond to this input, and the correlations $\langle x_i x_j \rangle_{\text{clamped}}$ are recorded. These correlations strengthen connections between units that tend to be co-active in response to real-world stimuli—a Hebbian process.

In the **sleep phase**, the network is disconnected from sensory input and generates its own "fantasies" by running freely from its equilibrium distribution. The correlations $\langle x_i x_j \rangle_{\text{free}}$ recorded during this phase are used to weaken connections that are spuriously co-active in the model's internal fantasies—an anti-Hebbian process that prevents the network from generating patterns not present in the data.

The difference between the wake and sleep statistics drives learning. This **contrastive Hebbian learning** elegantly combines local synaptic updates with a global objective.

## The Computational Bottleneck: Equilibrium Sampling
Despite its theoretical elegance, the classic Boltzmann machine suffers from a severe practical limitation: **the need to sample from the equilibrium distribution in both phases is computationally prohibitive**.

In the clamped phase, we must sample hidden states from $P(\mathbf{h} \mid \mathbf{v})$. Because the hidden units are fully connected to one another, they do not become conditionally independent given the visible units. Computing the exact conditional distribution requires summing over all $2^{N_h}$ hidden configurations, which is intractable for non-trivial $N_h$. Instead, we must run a Markov chain over the hidden units until it reaches equilibrium—a process that can be extremely slow, especially when the energy landscape has high barriers.

In the free phase, the situation is even worse: we must run a Markov chain over **all** $N$ units until the joint distribution reaches equilibrium. The mixing time of this chain scales poorly with network size, and for large networks, obtaining unbiased samples is effectively impossible.

## The Unconstrained Connectivity: A Double-Edged Sword
The fully connected architecture gives the classic Boltzmann machine its theoretical power: it is a **universal approximator** for probability distributions over binary vectors, given enough hidden units. The hidden-hidden connections allow the model to capture complex, higher-order dependencies that simpler models miss.

However, this same connectivity is the source of the computational intractability. The hidden units are not conditionally independent given the visible units, nor are the visible units conditionally independent given the hidden units. Every sampling step requires computing the total input to a unit from **all** other units, visible and hidden alike, and the Markov chain must mix over a state space of size $2^{N}$.

In practice, training a classic Boltzmann machine on anything but tiny toy problems was infeasible with 1980s and 1990s computing resources. The algorithm remained a theoretical curiosity until the introduction of architectural restrictions that dramatically simplified the sampling process.

## The Path Forward: Restricted Architectures
The limitations of the classic Boltzmann machine motivated a crucial architectural innovation: the **Restricted Boltzmann Machine (RBM)**. By removing all visible-visible and hidden-hidden connections, the RBM makes the units in one layer conditionally independent given the other layer. This seemingly small change has profound computational consequences:

- **Clamped phase sampling**: Given visible units, the hidden units become independent, and their states can be sampled in parallel in a single step. No Markov chain is needed.
- **Free phase sampling**: Block Gibbs sampling alternates between updating all hidden units given the visible units and all visible units given the hidden units. While still requiring multiple iterations to reach equilibrium, each iteration is fast and parallelizable.

The RBM, combined with the **contrastive divergence** approximation (Section [3.3 Contrastive Divergence](kpp-theoretical-foundations-ebms-cd.md)), transformed the Boltzmann machine from a theoretical construct into a practical tool for unsupervised feature learning. The classic Boltzmann machine thus serves as the conceptual foundation, the "Platonic ideal", from which more tractable variants are derived.

## Recap: The Classic Boltzmann Machine
The classic Boltzmann machine embodies the core principles of energy-based generative modeling:

- **Architecture**: A fully connected, symmetric recurrent network of stochastic binary units, partitioned into visible and hidden sets.
- **Energy function**: A quadratic form over all units, with weights and biases as parameters.
- **Dynamics**: Asynchronous stochastic updates that converge to the Boltzmann equilibrium distribution.
- **Learning**: Contrastive Hebbian updates driven by the difference between clamped (wake) and free-running (sleep) correlations.
- **Limitation**: The need for equilibrium sampling over a fully connected graph renders exact training computationally intractable for all but the smallest networks.

The next section introduces the **Restricted Boltzmann Machine**, which imposes structural constraints that dramatically reduce sampling complexity while retaining much of the expressive power of the classic model.

## Summary
- The **classic Boltzmann machine** is a fully connected, undirected probabilistic graphical model with symmetric weights and stochastic binary units.
- Units are divided into **visible** (observed data) and **hidden** (latent features) groups, with full connectivity among all units.
- The energy function defines a joint Boltzmann distribution, and stochastic updates guarantee convergence to equilibrium.
- Learning follows a **contrastive Hebbian rule**: increase weights based on correlations when visible units are clamped to data, and decrease weights based on correlations when the network runs freely.
- The fully connected architecture makes **exact inference and equilibrium sampling intractable**, limiting practical applications.
- The computational bottleneck of the classic Boltzmann machine motivated the development of the **Restricted Boltzmann Machine (RBM)**, which removes certain connections to enable efficient sampling.

