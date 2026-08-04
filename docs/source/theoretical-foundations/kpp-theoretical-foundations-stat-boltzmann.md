---
title: 1.2 The Boltzmann Distribution and Equilibrium
slug: kpp-theoretical-foundations-stat-boltzmann
sidebar_position: 1
hide: false
hide_child: false
---


# 1.2 The Boltzmann Distribution and Equilibrium

> Having established the analogy between magnetic spins and neural states, we now turn to the central probabilistic law that governs such systems at thermal equilibrium: the **Boltzmann distribution**. This distribution provides the mathematical foundation for treating neural networks as probabilistic generative models rather than merely deterministic function approximators.

## Thermal Equilibrium and the Canonical Ensemble
Consider a physical system in contact with a large heat-bath at absolute temperature $T$. The system constantly exchanges energy with the bath: microscopic fluctuations push it into higher-energy configurations, while the tendency to minimize energy pulls it back toward lower-energy states. After sufficient time, the system reaches **thermal equilibrium**, a condition in which macroscopic properties cease to change, even though microscopic configurations continue to fluctuate.

In the framework of statistical mechanics, an equilibrium system is described by the **canonical ensemble**. The fundamental postulate is that all microstates with the same energy are equally probable, and the probability of the system occupying a specific microstate $\mathbf{x}$ with energy $E(\mathbf{x})$ depends only on that energy and the temperature. This dependence is expressed by the Boltzmann distribution.

## The Boltzmann Distribution
The probability of finding the system in state $\mathbf{x}$ at temperature $T$ is:

$$P(\mathbf{x}) = \frac{1}{Z} \exp\left(-\frac{E(\mathbf{x})}{k_B T}\right)$$

where:

- $E(\mathbf{x})$ is the energy of configuration $\mathbf{x}$.
- $k_B$ is the Boltzmann constant, which bridges temperature and energy scales. 
- $T$ is the absolute temperature in Kelvin.
- $Z$ is the **partition function** (from German *Zustandssumme*, "sum over states").

The factor $\exp(-E(\mathbf{x})/k_B T)$ is called the **Boltzmann factor.** The negative sign in the exponent ensures that configurations with *lower energy* receive *higher probability*. The temperature $T$ modulates the steepness of this preference: at low temperatures, the distribution is sharply peaked around the lowest-energy states; at high temperatures, the distribution flattens, assigning non-negligible probability to a much broader range of configurations.

## Statistical Physics Foundations: Derivation from the Microcanonical Ensemble
The Boltzmann distribution is not an arbitrary choice; it follows directly from the microcanonical ensemble when a small system is placed in contact with a large heat reservoir.

Consider a small system $\mathcal{S}$ with energy eigenstates $|i\rangle$ and energies $E_i$, in thermal contact with a large reservoir $\mathcal{R}$ at temperature $T$. The combined system $\mathcal{S} + \mathcal{R}$ is isolated, with total energy $E_{\text{total}}$. By the fundamental assumption of the microcanonical ensemble, all microstates of the combined system with energy in the range $(E_{\text{total}}, E_{\text{total}} + \delta E)$ are equally probable.

The number of microstates of the reservoir when $\mathcal{S}$ is in state $|i\rangle$ is:

$$\Omega_R(E_{\text{total}} - E_i) = \exp\left[ k^{-1} S_R(E_{\text{total}} - E_i) \right]$$

Since $E_i \ll E_{\text{total}}$, we expand the entropy of the reservoir:

$$S_R(E_{\text{total}} - E_i) \approx S_R(E_{\text{total}}) - \left( \frac{\partial S_R}{\partial E} \right) E_i = S_R(E_{\text{total}}) - \frac{1}{T} E_i$$

where we have used the thermodynamic definition of temperature:

$$\frac{1}{T} = \left( \frac{\partial S}{\partial E} \right)_{V,N}$$

Thus, the probability that $\mathcal{S}$ is in state $|i\rangle$ is proportional to the number of reservoir microstates:

$$P_i \propto \Omega_R(E_{\text{total}} - E_i) \propto \exp\left( -\frac{E_i}{k_B T} \right)$$

Normalizing by the partition function $Z = \sum_i \exp(-E_i / k_B T)$ yields the Boltzmann distribution.

This derivation reveals a crucial insight: **the temperature** $T$ **and the thermal noise in the Boltzmann machine are not arbitrary heuristics; they emerge from the coupling of the neural system to an external heat bath.** Stochasticity is a physical necessity for maintaining thermal equilibrium.

## The Partition Function: A Computational Bottleneck
The partition function $Z$ is defined as the sum of Boltzmann factors over all possible states:

$$Z = \sum_{\tilde{\mathbf{x}}} \exp\left(-\frac{E(\tilde{\mathbf{x}})}{k_B T}\right)$$

Its role is purely normalizing, it ensures that the probabilities over all states sum to one:

$$\sum_{\tilde{\mathbf{x}}} P(\tilde{\mathbf{x}}) = \frac{1}{Z} \sum_{\tilde{\mathbf{x}}} \exp\left(-\frac{E(\tilde{\mathbf{x}})}{k_B T}\right) = 1$$

While mathematically simple in definition, the partition function is computationally formidable. For a system of $N$binary units, the sum runs over $2^N$ distinct configurations. For even modest $N$(say, $N=100$, this number exceeds the estimated number of atoms in the observable universe. **The intractability of the partition function is the central computational challenge** that shapes every aspect of Boltzmann machine learning, from training algorithms to architectural constraints.

## The Thermodynamic Limit and Ensemble Equivalence
For a macroscopic system (large $N$), the canonical and microcanonical ensembles become equivalent. This is because the energy fluctuations in the canonical ensemble scale as:

$$\frac{\Delta E}{\langle E \rangle} \propto \frac{1}{\sqrt{N}}$$

which becomes negligible as $N \to \infty$. This is the **thermodynamic limit**. In this limit, the energy of the system is essentially fixed, and the canonical ensemble gives the same results as the microcanonical ensemble.

This property is important for Boltzmann machines: when the number of units is large, the model's equilibrium distribution is sharply peaked around the mean energy, and the system behaves as if it were isolated with a fixed energy.

## The Boltzmann Distribution in Neural Networks
Translating the physical expression into the language of neural networks requires only a redefinition of variables. For a Boltzmann machine with binary units $\mathbf{x} \in \{0,1\}^N$ (or equivalently $\{-1,+1\}^N$), we absorb the Boltzmann constant into the temperature parameter and define the **energy function** $E_\theta(\mathbf{x})$ parameterized by biases $\mathbf{b}$ and symmetric weights $\mathbf{W}$:

$$P_\theta(\mathbf{x}) = \frac{1}{Z_\theta} \exp\left(-E_\theta(\mathbf{x})\right)$$

where the partition function becomes:

$$Z_\theta = \sum_{\tilde{\mathbf{x}}} \exp\left(-E_\theta(\tilde{\mathbf{x}})\right)$$

and the energy is:

$$E_\theta(\mathbf{x}) = -\sum_i b_i x_i - \sum_{i<j} w_{ij} x_i x_j$$

Here we have set $k_B T = 1$ for notational simplicity; the temperature can be reintroduced as a scaling factor on the energy or, equivalently, as an inverse temperature parameter $\beta = 1/T$. In practice, the temperature is often treated as a hyperparameter controlling the stochasticity of sampling.

## The Meaning of Equilibrium in a Neural Context
When we say a Boltzmann machine samples from its **equilibrium distribution**, we mean that after running the stochastic update dynamics for a sufficiently long time, the probability of observing any particular configuration $\mathbf{x}$ converges to $P_\theta(\mathbf{x})$. This property is guaranteed by the fact that the update rule satisfies **detailed balance,** a condition ensuring that, in equilibrium, the flow of probability from state $\mathbf{x}$ to state $\mathbf{x}'$ exactly balances the reverse flow:

$$P_\theta(\mathbf{x}) \, P(\mathbf{x} \to \mathbf{x}') = P_\theta(\mathbf{x}') \, P(\mathbf{x}' \to \mathbf{x})$$

The stochastic neuron update rule introduced earlier,

$$P(x_i = 1) = \sigma\left( \sum_j w_{ij} x_j + b_i \right)$$

where $\sigma(z) = 1/(1+\exp(-z))$, is precisely designed to satisfy detailed balance with respect to the energy function $E_\theta$. Consequently, if we run the network long enough, which allowing it to "forget" its initial state, the configurations it generates will be distributed according to $P_\theta$.

## From Physics to Learning
The Boltzmann distribution reframes the learning problem in probabilistic terms. Instead of asking "how do we compute a function from inputs to outputs?", we ask **"how do we adjust the parameters** $\theta$ **so that the equilibrium distribution** $P_\theta$ **matches the empirical distribution of the observed data?"** This shift in perspective is profound. The energy function $E_\theta(\mathbf{x})$ becomes a **scoring function**: data points that are typical of the training set should be assigned low energy (high probability), while atypical or impossible configurations should reside at high energy (low probability). Learning, then, consists of **sculpting the energy landscape** by adjusting the weights and biases so that the valleys of low energy coincide with regions of high data density.

The challenge, as we shall see, lies in estimating how the partition function $Z_\theta$ changes as we modify the parameters. Because $Z_\theta$ depends on all $2^N$ states, we cannot compute it, or its gradient exactly. The remainder of this tutorial series is, in large measure, a study of how to circumvent this intractability through clever sampling techniques and architectural restrictions.

## Summary
- The **Boltzmann distribution** $P(\mathbf{x}) \propto \exp(-E(\mathbf{x})/T)$emerges naturally at thermal equilibrium and forms the probabilistic core of the Boltzmann machine.
- The **partition function** $Z$normalizes the distribution but requires summing over an exponential number of states, making exact computation infeasible for large systems.
- In neural terms, **equilibrium** means the network's stochastic dynamics have converged to a stationary distribution defined by its weights and biases.
- Learning in a Boltzmann machine is equivalent to **energy landscape shaping**: lowering the energy of observed data configurations while implicitly raising the energy elsewhere.
- The intractability of $Z$ motivates the development of approximate inference and learning algorithms, which will be explored in subsequent sections.

