---
title: 3.2 The Intractable Partition Function Problem
slug: kpp-theoretical-foundations-ebms-partition
sidebar_position: 8
hide: false
hide_child: false
---


# 3.2 The Intractable Partition Function Problem

> Section [3.1 Defining the Objective: Low Energy for Real Data](kpp-theoretical-foundations-ebms-def.md) introduced the maximum likelihood objective for energy-based models and revealed that the gradient of the negative log-likelihood contains a term involving an expectation under the model distribution. This expectation, in turn, requires summing over all possible configurations, a sum that is encoded in the **partition function** $Z_\theta$. In this section, we examine why the partition function is computationally intractable, what consequences this intractability has for learning and inference, and why it constitutes the central algorithmic challenge for Boltzmann machines and energy-based models more broadly.

## The Partition Function: Definition and Its Implications
Recall the definition of the partition function for a Boltzmann machine with binary units $\mathbf{x} \in \{0,1\}^N$:

$$Z_\theta = \sum_{\tilde{\mathbf{x}} \in \{0,1\}^N} \exp\left(-E_\theta(\tilde{\mathbf{x}})\right)$$

The sum runs over **all** $2^N$ **possible binary vectors** of length $N$. For a modest network of $N=100$ visible units (a tiny image by modern standards, only $10 \times 10$ pixels), the number of terms in this sum is:

$$2^{100} \approx 1.27 \times 10^{30}$$

To put this number in perspective:

- The estimated number of stars in the observable universe is about $10^{22}$ to $10^{24}$.
- The age of the universe is approximately $4.35 \times 10^{17}$ seconds.
- Even if we could evaluate one trillion ($10^{12}$) configurations per second, enumerating $2^{100}$ configurations would take over $10^{10}$times the age of the universe.

And $N = 100$ is **tiny**. Real-world Boltzmann machines and restricted Boltzmann machines often have hundreds of visible units and hundreds or thousands of hidden units, yielding state spaces of size $2^{N_{\text{vis}} + N_{\text{hid}}}$. The partition function is **exponentially large** in the number of units.

## Why the Partition Function Matters
The partition function appears in three critical contexts:

1. **Probability Evaluation** To compute the probability $P_\theta(\mathbf{x})$ of a specific configuration, we must divide the Boltzmann factor by $Z_\theta$:

$$P_\theta(\mathbf{x}) = \frac{\exp(-E_\theta(\mathbf{x}))}{Z_\theta}$$

Without $Z_\theta$, we have only an unnormalized score. This means we cannot directly compute likelihoods on test data, making model comparison and hyperparameter tuning difficult.

1. **Learning by Gradient Descent** Recall from Section [3.1 Defining the Objective: Low Energy for Real Data](kpp-theoretical-foundations-ebms-def.md) that the gradient of the negative log-likelihood decomposes into a data term and a model term:

$$\nabla f(\theta) = -\mathbb{E}_{\text{data}} \left[ \frac{\partial E_\theta}{\partial \theta} \right] + \mathbb{E}_{\text{model}} \left[ \frac{\partial E_\theta}{\partial \theta} \right]$$

> Note: $\nabla f(\theta) = -\frac{\partial \mathcal{L}}{\partial \theta}$, where $\mathcal{L}$ is the negative log-likelihood minimized in many textbooks.

The first term is straightforward to estimate from the training set. The second term, i.e. the expectation under the model distribution $P_\theta$, is the source of difficulty. As shown in the expression above, computing this expectation exactly requires summing over all $2^N$ possible configurations, weighted by the Boltzmann factor. Both the numerator and the denominator involve the partition function $Z_\theta$, making exact gradient evaluation computationally intractable for any non-trivial model.

The **model expectation** $\mathbb{E}_{\text{model}}[\cdot]$*is an average over the model distribution*$P_\theta$:

$$\mathbb{E}_{\text{model}} \left[ \frac{\partial E_\theta}{\partial \theta} \right] = \sum_{\tilde{\mathbf{x}}} P_\theta(\tilde{\mathbf{x}}) \frac{\partial E_\theta(\tilde{\mathbf{x}})}{\partial \theta} = \frac{1}{Z_\theta} \sum_{\tilde{\mathbf{x}}} \exp(-E_\theta(\tilde{\mathbf{x}})) \frac{\partial E_\theta(\tilde{\mathbf{x}})}{\partial \theta}$$

Both the numerator and the denominator involve sums over all $2^N$ configurations. Exact gradient computation is infeasible.

1. **Sampling and Inference** Generating samples from the model distribution typically requires Markov chain Monte Carlo (MCMC) methods, such as Gibbs sampling. While MCMC does not require computing $Z_\theta$ explicitly, it does require running the Markov chain long enough to reach equilibrium. The time required to obtain independent samples scales with the **mixing time** of the chain, which can be exponentially long in the system size, especially when the energy landscape has high barriers.

## The Partition Function as a Free Energy Barrier
In statistical physics, the partition function is intimately related to the **Helmholtz free energy** $F = -\log Z$.

(Setting $ k_B T = 1$ for notational simplicity.) The free energy summarizes the balance between energy and entropy. For a given visible configuration $\mathbf{v}$ in a Boltzmann machine with hidden units $\mathbf{h}$, the **free energy** is defined analogously as $F_\theta(\mathbf{v}) = -\log \sum_{\tilde{\mathbf{h}}} \exp\left(-E_\theta(\mathbf{v}, \tilde{\mathbf{h}})\right)$.

The probability of a visible vector is then:

$P_\theta(\mathbf{v}) = \frac{\exp(-F_\theta(\mathbf{v}))}{\sum_{\tilde{\mathbf{v}}} \exp(-F_\theta(\tilde{\mathbf{v}}))}$.

Even with hidden units marginalized out, the denominator still sums over all $2^{N_{\text{vis}}}$ visible configurations. The intractability remains.

## The Consequence: Exact Maximum Likelihood Is Impossible
The intractability of $Z_\theta$ has a stark implication: **exact maximum likelihood learning of a Boltzmann machine is computationally impossible for all but the smallest toy problems.** Any practical training algorithm must rely on approximations.

This is not merely an engineering inconvenience; it is a fundamental computational barrier. The partition function sum is a prototypical #P-complete problem in computational complexity theory, meaning it is at least as hard as counting the number of satisfying assignments to a Boolean formula.

## Hessian and Non-Convexity

The difficulty extends beyond the gradient. The Hessian of the log-likelihood function also reveals the complexity of the optimization landscape. For a Boltzmann machine with only visible units, the Hessian has a simple and revealing form:

$$\nabla^2 f(\theta) = -\text{Cov}_\theta(\mathbf{S})$$

where $\mathbf{S}$ is the vector of sufficient statistics (e.g., $X_i$ and $X_i X_j$). This means the Hessian is negative definite, and the optimization is **concave**—there are no non-global local minima when all units are visible.

However, when hidden units are introduced, the Hessian becomes:

$$\nabla^2 f(\theta) = \mathbb{E}_{\text{target}}[\text{Cov}_\theta(\mathbf{S}|\mathbf{X})] - \text{Cov}_\theta(\mathbf{S})$$

The first term is the conditional covariance given the visible units, and the second term is the unconditional covariance under the model. This expression is **not** necessarily negative semidefinite, meaning that **the log-likelihood is not concave** in the presence of hidden units. Consequently, gradient-based methods can converge to local optima, and the quality of the learned model depends crucially on the initialization and optimization algorithm.

## Approaches to Taming the Intractability
Faced with this barrier, researchers have developed several families of approximation strategies. Understanding these strategies is essential for navigating the landscape of energy-based learning.

1. **Sampling-Based Approximations** Instead of summing over all configurations, we can **sample** configurations from the model distribution using MCMC. The model expectation is then approximated by an empirical average over the samples:

$$\mathbb{E}_{\text{model}} \left[ \frac{\partial E_\theta}{\partial \theta} \right] \approx \frac{1}{K} \sum_{k=1}^K \frac{\partial E_\theta(\tilde{\mathbf{x}}^{(k)})}{\partial \theta}, \quad \tilde{\mathbf{x}}^{(k)} \sim P_\theta$$

The challenge is that obtaining unbiased samples requires running the Markov chain to equilibrium for every gradient step, which is prohibitively slow. **Contrastive divergence** (Section [3.3 Contrastive Divergence](kpp-theoretical-foundations-ebms-cd.md)) and **persistent contrastive divergence** address this by using short MCMC runs initialized from the data or from previous model states.

1. **Variational Approximations** Variational methods replace the intractable model distribution $P_\theta$ with a simpler, tractable **variational distribution** $Q_\phi$ from a restricted family. The parameters $ \phi$ are optimized to make $Q_\phi$ as close as possible to $P_\theta$ in the sense of Kullback-Leibler divergence. This transforms the learning problem into an optimization over both $ \theta$ and $\phi$.

While variational methods provide a rigorous lower bound on the log-likelihood, they often require making strong independence assumptions (e.g., mean-field approximations) that can limit the model's expressive power.

1. **Architectural Constraints** By restricting the connectivity pattern of the Boltzmann machine, we can make certain computations tractable. The **Restricted Boltzmann Machine (RBM)** is the most celebrated example. By eliminating visible-visible and hidden-hidden connections, the conditional distributions $P(\mathbf{h} \mid \mathbf{v})$ and $P(\mathbf{v} \mid \mathbf{h})$ factorize into products of independent Bernoulli distributions. This allows efficient Gibbs sampling and enables the contrastive divergence algorithm.

1. **Score Matching and Noise Contrastive Estimation** These are alternative training criteria that circumvent the partition function entirely. **Score matching** minimizes the Fisher divergence between the model and data distributions by matching the gradients of their log-densities, which does not require $Z_\theta$. **Noise contrastive estimation (NCE)** transforms the unsupervised learning problem into a supervised logistic regression task of distinguishing data samples from noise samples, where the partition function becomes a learnable parameter.

1. **Annealed Importance Sampling** For model evaluation (rather than training), **annealed importance sampling (AIS)** provides an unbiased estimate of $Z_\theta$. It works by gradually transforming a simple base distribution (e.g., uniform) into the model distribution through a sequence of intermediate distributions, using importance weights to correct for the sampling bias. While computationally expensive, AIS is a gold standard for evaluating generative models when exact likelihoods are required.

## The Partition Function in the Era of Deep Learning
It is worth noting that the intractability of the partition function is not unique to Boltzmann machines. Many modern deep generative models face analogous challenges:

- **Generative Adversarial Networks (GANs)** avoid the partition function entirely by using an adversarial training criterion that does not require likelihood evaluation.
- **Variational Autoencoders (VAEs)** use a variational lower bound that circumvents the partition function.
- **Normalizing Flows** design architectures where the Jacobian determinant, the analog of $Z_\theta$, is tractable by construction.
- **Diffusion Models** learn to reverse a gradual noising process, with a training objective that involves only tractable conditional distributions.

The Boltzmann machine's struggle with the partition function thus foreshadows a central theme in deep generative modeling: the tension between expressive power and tractable normalization.

## Summary
- The **partition function** $Z_\theta = \sum_{\mathbf{x}} \exp(-E_\theta(\mathbf{x}))$ is a sum over an exponential number of configurations, making exact computation infeasible for non-trivial models.
- The intractability of $Z_\theta$ prevents exact **probability evaluation**, exact **gradient computation**, and efficient exact **sampling**.
- This computational barrier implies that **exact maximum likelihood learning is impossible** for general Boltzmann machines.
- Approximation strategies fall into several categories: **sampling-based** (MCMC, contrastive divergence), **variational** (mean-field), **architectural constraints** (RBM), and **alternative objectives** (score matching, NCE).
- The partition function problem is a fundamental challenge that connects Boltzmann machines to contemporary deep generative models, many of which are designed explicitly to circumvent or tame this intractability.
- The **Restricted Boltzmann Machine** and **contrastive divergence,** topics of the upcoming sections, provide a pragmatic and influential solution to this problem for a restricted but powerful class of architectures.
