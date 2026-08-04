---
title: '3.1 Defining the Objective: Low Energy for Real Data'
slug: kpp-theoretical-foundations-ebms-def
sidebar_position: 7
hide: false
hide_child: false
---


# 3.1 Defining the Objective: Low Energy for Real Data

> The preceding chapters have introduced two essential perspectives: the **probabilistic framework** of statistical physics (Chapter 1) and the **architectural principles** of recurrent neural networks (Chapter 2). We now unite these perspectives under the umbrella of **Energy-Based Models (EBMs)**. This section defines the core learning objective that drives all energy-based models, including the Boltzmann machine: to shape the energy landscape so that real data resides in deep valleys, while all other configurations occupy higher ground.

## The Energy-Based Modeling Paradigm
An energy-based model is defined by a scalar **energy function** $E_\theta(\mathbf{x})$, parameterized by $ \theta$, that assigns a real-valued energy to every possible configuration $\mathbf{x}$ of the variables of interest. For a Boltzmann machine, $\mathbf{x}$ represents the joint state of visible and hidden units, but the framework applies broadly to any structured configuration.

The energy function induces a probability distribution via the **Boltzmann distribution** (or Gibbs distribution):

$$P_\theta(\mathbf{x}) = \frac{\exp\left(-E_\theta(\mathbf{x})\right)}{Z_\theta}, \quad \text{where} \quad Z_\theta = \sum_{\tilde{\mathbf{x}}} \exp\left(-E_\theta(\tilde{\mathbf{x}})\right)$$

This formulation accomplishes a fundamental transformation: **low energy corresponds to high probability**. The model assigns higher likelihood to configurations that it deems "good" or "compatible" according to its internal criteria.

Crucially, the energy function is **unnormalized,** it provides a relative ordering of configurations but does not directly yield probabilities without the partition function $Z_\theta$. This is both a strength (flexibility in designing energy functions) and a challenge (intractability of $Z_\theta$.

## The Learning Objective: Maximum Likelihood and KL Divergence
Given a training dataset $\mathcal{D} = \{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots, \mathbf{x}^{(M)}\}$ drawn independently from an unknown data distribution $P_{\text{data}}(\mathbf{x})$, the goal of learning is to adjust the parameters $ \theta$ so that the model distribution $P_\theta(\mathbf{x})$ approximates $P_{\text{data}}$ as closely as possible.

The standard criterion for this approximation is **maximum likelihood estimation (MLE)**. We seek parameters that maximize the probability of the observed data under the model:

$$\theta^* = \arg\max_\theta \prod_{m=1}^M P_\theta(\mathbf{x}^{(m)}) = \arg\max_\theta \sum_{m=1}^M \log P_\theta(\mathbf{x}^{(m)})$$

Equivalently, we minimize the **negative log-likelihood**:

$$\mathcal{L}(\theta) = -\frac{1}{M} \sum_{m=1}^M \log P_\theta(\mathbf{x}^{(m)})$$

Substituting the Boltzmann distribution form of $P_\theta$:

$$\log P_\theta(\mathbf{x}) = -E_\theta(\mathbf{x}) - \log Z_\theta$$

Thus, the negative log-likelihood becomes:

$$\mathcal{L}(\theta) = \frac{1}{M} \sum_{m=1}^M E_\theta(\mathbf{x}^{(m)}) + \log Z_\theta$$

This decomposition reveals the two competing forces that shape learning:

1. **Data term**: $\frac{1}{M} \sum_{m} E_\theta(\mathbf{x}^{(m)})$, the average energy of real data points. Minimizing this term pushes the energy of observed data **downward**.
2. **Partition function term**: $\log Z_\theta$, the log of the sum of exponentials of negative energies over all configurations. Minimizing this term pushes the energy of **all** configurations, and especially those with currently low energy, **upward**, because $\log Z_\theta$ grows as more configurations acquire low energy.

The learning process is therefore a **tug-of-war**: the data term pulls down the energy at the observed data points, while the partition function term pushes up the energy everywhere else to maintain normalization. The equilibrium of this tug-of-war is reached when the model distribution matches the data distribution.

## Connection to KL Divergence

The MLE objective has a deeper interpretation from information theory. Minimizing the negative log-likelihood is **equivalent** to minimizing the **Kullback-Leibler (KL) divergence** from the data distribution to the model distribution:

$$\text{KL}(P_{\text{data}} \parallel P_\theta) = \sum_{\tilde{\mathbf{x}}} P_{\text{data}}(\tilde{\mathbf{x}}) \log \frac{P_{\text{data}}(\tilde{\mathbf{x}})}{P_\theta(\tilde{\mathbf{x}})}$$

This can be rewritten as:

$$\text{KL}(P_{\text{data}} \parallel P_\theta) = -\mathbb{E}_{\text{data}}[\log P_\theta(\mathbf{X})] - H(P_{\text{data}})$$

where $H(P_{\text{data}}) = -\mathbb{E}_{\text{data}}[\log P_{\text{data}}(\mathbf{X})]$ is the entropy of the data distribution (a constant independent of $\theta$). Therefore, minimizing $\text{KL}(P_{\text{data}} \parallel P_\theta)$ is equivalent to maximizing the expected log-likelihood $f(\theta) = \mathbb{E}_{\text{data}}[\log P_\theta(\mathbf{X})] = \frac{1}{M} \sum_{m=1}^M \log P_\theta(\mathbf{x}^{(m)})$.

This equivalence reveals that **MLE is fundamentally about matching distributions,** which means the model distribution $P_\theta$ is trained to be as close as possible to the empirical data distribution in the sense of KL divergence.

## Gradient of the Log-Likelihood

To perform gradient-based optimization, we compute the gradient of the log-likelihood $f(\theta)$ with respect to the parameters:

$$\nabla f(\theta) = \mathbb{E}_{\text{data}}[\nabla \log P_\theta(\mathbf{X})] = -\mathbb{E}_{\text{data}}[\nabla E_\theta(\mathbf{X})] - \nabla \log Z_\theta$$

The gradient of $\log Z_\theta$ is:

$$\nabla \log Z_\theta = \frac{1}{Z_\theta} \sum_{\tilde{\mathbf{x}}} \exp(-E_\theta(\tilde{\mathbf{x}})) \nabla E_\theta(\tilde{\mathbf{x}}) = \mathbb{E}_\theta[\nabla E_\theta(\mathbf{X})]$$

Thus, the gradient of the log-likelihood becomes:

$$\nabla f(\theta) = -\mathbb{E}_{\text{data}}[\nabla E_\theta(\mathbf{X})] + \mathbb{E}_\theta[\nabla E_\theta(\mathbf{X})]$$

This contrastive form, i.e. the difference between an expectation under the **data distribution** and an expectation under the **model distribution,** is the mathematical heart of all Boltzmann machine learning algorithms. The first term lowers the energy of observed data configurations; the second term raises the energy of configurations that the model currently deems probable.

## Special Case: Visible-Only Boltzmann Machine

For a Boltzmann machine with only visible units (no hidden units), the gradient simplifies to a particularly transparent form. Substituting the energy function $E_\theta(\mathbf{x}) = -\mathbf{b}^\top \mathbf{x} - \frac{1}{2} \mathbf{x}^\top \mathbf{W} \mathbf{x}$ and taking the derivative with respect to a weight $w_{ij}$ yields:

$$\frac{\partial f(\theta)}{\partial w_{ij}} = \langle X_i X_j \rangle_{\text{data}} - \langle X_i X_j \rangle_{\text{model}}$$

where $\langle X_i X_j \rangle_{\text{data}}$ is the expected product of the activities of units $i$ and $j$ when the visible units are clamped to the data (the positive phase), and $\langle X_i X_j \rangle_{\text{model}}$ is the expected product when the network samples freely from its equilibrium distribution (the negative phase). This simple expression directly captures the essence of contrastive learning: co-activations observed in the data are reinforced, while those generated by the model’s own fantasies are suppressed.

## Geometric Intuition: Sculpting the Energy Landscape
Imagine the energy landscape as a deformable surface over the space of all configurations. Initially, the surface may be flat (all configurations have similar energy and probability) or arbitrarily wrinkled.

Learning proceeds by iteratively applying two operations:

- **Excavation**: At the locations of training examples, we dig downward, which lowering the energy to create basins of attraction.
- **Elevation**: Everywhere else, we raise the landscape to prevent the model from assigning high probability to regions where no data resides.

The **gradient** of the log-likelihood with respect to the parameters $\theta$formalizes this intuition:

$$\frac{\partial f}{\partial \theta} = -\frac{1}{M} \sum_{m=1}^M \frac{\partial E_\theta(\mathbf{x}^{(m)})}{\partial \theta} + \mathbb{E}_{P_\theta} \left[ \frac{\partial E_\theta(\mathbf{x})}{\partial \theta} \right]$$

The first term is the negative average gradient of the energy at the data points, the direction of steepest descent for lowering data energy. The second term is the expected gradient of the energy under the model's own distribution, the direction that would lower the energy of configurations the model currently favors.

The positive sign on the second term means that, overall, we move parameters in the direction that **increases energy** where the model currently puts probability mass. This is the mathematical expression of tug-of-war.

>**Note on notation**: In many textbooks, the negative log-likelihood $\mathcal{L}(\theta)$ is minimized. Here we maximize $f(\theta) = -\mathcal{L}(\theta)$ to keep the gradient sign consistent with the contrastive divergence algorithm introduced in Section [3.3 Contrastive Divergence](kpp-theoretical-foundations-ebms-cd.md).

## Why Low Energy for Real Data?
The mandate "low energy for real data" is not an arbitrary choice; it follows directly from the maximum likelihood principle and the exponential link between energy and probability. But beyond mathematics, this objective carries several intuitive and practical advantages:

1. **Unnormalized Flexibility**: The energy function can be designed without concern for normalization. Any differentiable function of the configuration can serve as an energy. This allows incorporating domain knowledge, constraints, and structured architectures.
2. **Natural Handling of Uncertainty**: In regions where no data has been observed, the model's energy can remain high, corresponding to low probability. The model is not forced to make arbitrary predictions in unobserved regions.
3. **Generation by Descent**: Once trained, new samples can be generated by starting from a random configuration and following a stochastic descent procedure on the energy landscape. This yields configurations that are typical under the learned distribution.
4. **Compositionality**: Energy functions can be combined additively. If $E_1(\mathbf{x})$ and $E_2(\mathbf{x})$ model different aspects of the data, the sum $E_1 + E_2$ defines a product of experts model that captures both constraints.
5. **Physical Plausibility**: The energy-based view connects learning to well-understood physical processes like annealing and phase transitions, providing a rich conceptual vocabulary for understanding model behavior.

## The Challenge: The Partition Function Gradient
The elegance of the energy-based objective comes at a computational cost. The gradient involves an expectation under the model distribution:

$$\mathbb{E}_{P_\theta} \left[ \frac{\partial E_\theta(\mathbf{x})}{\partial \theta} \right] = \sum_{\tilde{\mathbf{x}}} P_\theta(\tilde{\mathbf{x}}) \frac{\partial E_\theta(\tilde{\mathbf{x}})}{\partial \theta}$$

This sum over all possible configurations is intractable for any non-trivial model. We cannot compute it exactly, nor can we enumerate all configurations to approximate it.

This intractability is the central computational challenge that defines the field of energy-based learning. Every practical algorithm for training Boltzmann machines can be understood as a strategy for approximating this model expectation.

The subsequent sections will explore these strategies in depth:

- **Section** [3.2 The Intractable Partition Function Problem](kpp-theoretical-foundations-ebms-partition.md) formalizes the partition function problem and its implications.
- **Section** [3.3 Contrastive Divergence](kpp-theoretical-foundations-ebms-cd.md) introduces **contrastive divergence**, the breakthrough approximation that made Boltzmann machines practical.
- **Chapters 4 and beyond** develop specific architectures (Restricted Boltzmann Machines) and training algorithms that exploit structural constraints to make the model expectation tractable.

## Connection to Discriminative Learning
It is instructive to contrast the energy-based generative objective with the more familiar **discriminative** objective used in feedforward classifiers.

In a discriminative model, such as a logistic regression or a multilayer perceptron classifier, we model the conditional distribution $P(y \mid \mathbf{x})$ directly, without modeling the input distribution $P(\mathbf{x})$. The energy function is defined over the joint space $(\mathbf{x}, y)$, and the probability of a label given an input is:

$$P(y \mid \mathbf{x}) = \frac{\exp(-E(\mathbf{x}, y))}{\sum_{y'} \exp(-E(\mathbf{x}, y'))}$$

Crucially, the partition function now sums only over the possible labels $y'$ for a fixed input $\mathbf{x}$. This sum is typically tractable (e.g., over a finite set of classes). The intractable sum over all configurations $\mathbf{x}$ is **not required** for discriminative training.

This observation explains why discriminative models have been so successful in supervised learning: they circumvent the partition function bottleneck entirely. However, they pay a price: they cannot generate new inputs, complete missing data, or learn from unlabeled examples.

Energy-based generative models embrace the full complexity of the partition function because they aim to model the entire data distribution, not just the decision boundary between classes. The reward is a richer, more flexible model capable of unsupervised learning, generation, and inference in any direction.

## Summary
- **Energy-Based Models** define a probability distribution via $P_\theta(\mathbf{x}) \propto \exp(-E_\theta(\mathbf{x}))$, linking low energy to high probability.
- The learning objective is **maximum likelihood**, which decomposes into minimizing the average energy of data while controlling the partition function.
- The gradient of the log-likelihood $f(\theta)$ involves a **data term** (easy to compute) and a **model expectation** (intractable to compute exactly).
- Learning is geometrically interpreted as **sculpting the energy landscape**: lowering basins at data points while raising the surrounding terrain.
- The intractability of the model expectation is the central computational challenge that motivates approximate methods like **contrastive divergence**.
- Generative energy-based models contrast with discriminative models by modeling the full joint distribution, enabling generation and unsupervised learning at the cost of the partition function intractability.
