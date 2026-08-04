---
title: 4.2 Restricted Boltzmann Machine (RBM)
slug: kpp-theoretical-foundations-bm-restricted
sidebar_position: 11
hide: false
hide_child: false
---


# 4.2 Restricted Boltzmann Machine (RBM)

> The classic Boltzmann machine, as presented in Section [4.1 The Classic Boltzmann Machine: Visible and Hidden Symmetry](kpp-theoretical-foundations-bm-overall.md), is theoretically powerful but computationally intractable due to the need for equilibrium sampling over a fully connected graph. The **Restricted Boltzmann Machine (RBM)**, introduced by Paul Smolensky in 1986 under the name *Harmonium* and later popularized by Geoffrey Hinton, imposes a simple but profound architectural constraint that dramatically reduces sampling complexity while preserving the ability to model complex probability distributions. This section details the RBM architecture, its mathematical properties, and why it has become the workhorse of energy-based unsupervised learning.

## The Architectural Restriction: A Bipartite Graph
The defining characteristic of an RBM is its **restricted connectivity**: the network is organized as a **bipartite graph** with two layers, a visible layer and a hidden layer. Connections exist **only** between visible and hidden units. There are **no visible-visible connections** and **no hidden-hidden connections**.

Formally, let the visible units be $\mathbf{v} = (v_1, v_2, \ldots, v_{N_v})$ with biases $\mathbf{b} = (b_1, \ldots, b_{N_v})$, and the hidden units be $\mathbf{h} = (h_1, h_2, \ldots, h_{N_h})$ with biases $\mathbf{c} = (c_1, \ldots, c_{N_h})$. The weight matrix $\mathbf{W}$ is of size $N_v \times N_h$, where $w_{ij}$ connects visible unit $i$ to hidden unit $j$. There is no weight matrix for visible-visible or hidden-hidden pairs.

This restricted topology is illustrated below:

This restricted topology yields a crucial property: **conditional independence**. Given the visible units, the hidden units become independent of each other. Conversely, given the hidden units, the visible units become independent.

## Energy Function and Joint Distribution
The energy of a joint configuration $(\mathbf{v}, \mathbf{h})$ in an RBM with binary visible and binary hidden units is:

$$E(\mathbf{v}, \mathbf{h}) = -\sum_{i=1}^{N_v} b_i v_i - \sum_{j=1}^{N_h} c_j h_j - \sum_{i=1}^{N_v} \sum_{j=1}^{N_h} v_i w_{ij} h_j$$

In matrix notation:

$$E(\mathbf{v}, \mathbf{h}) = -\mathbf{b}^\top \mathbf{v} - \mathbf{c}^\top \mathbf{h} - \mathbf{v}^\top \mathbf{W} \mathbf{h}$$

The joint probability distribution is, as always, given by the Boltzmann distribution:

$$P(\mathbf{v}, \mathbf{h}) = \frac{1}{Z} \exp\left(-E(\mathbf{v}, \mathbf{h})\right)$$

with the partition function:

$$Z = \sum_{\tilde{\mathbf{v}}, \tilde{\mathbf{h}}} \exp\left(-E(\tilde{\mathbf{v}}, \tilde{\mathbf{h}})\right)$$

## Probability Distributions in RBMs
To better understand the generative and training processes of an RBM, we need to clearly distinguish several key probability distributions. All of them originate from the joint distribution $p(v, h)$, but they differ in their normalization.

1. **Joint distribution** $p(v, h)$:

$$p(v, h) = \frac{1}{Z} \exp(-E(v, h))$$

where $Z = \sum_{v,h} \exp(-E(v, h))$. This is the most complete description of the model.

1. **Marginal distribution** $p(v)$:

This is the distribution we care most about. Because it directly defines the probability that the RBM assigns to observed visible data (e.g., images, features) without reference to the hidden units. During training, our goal (maximum likelihood estimation) is to maximize $p(\mathbf{v})$ over the training data. A well-trained RBM should be able to sample new visible configurations from $p(\mathbf{v})$ that look like real data. The hidden units are latent auxiliary variables that capture complex dependencies, but $p(\mathbf{v})$ is the ultimate evaluation criterion for the generative quality.

It is obtained by summing over all possible hidden unit states:

$$p(v) = \frac{\sum_{h} \exp(-E(v, h))}{Z}$$

This is the distribution we care most about because it directly defines the probability that the RBM assigns to observed visible data (e.g., images, features) without reference to the hidden units. During training, our goal is to maximize $p(v)$ over the training data. Ideally, a well-trained RBM should be able to sample new visible configurations from $p(v)$ that look like real data. The hidden units only serve as latent auxiliary variables to capture complex dependencies, but $p(v)$ is the ultimate evaluation criterion for the generative quality.

1. **Conditional probabilities**:

These are the key to the computational efficiency of RBMs. Due to the bipartite structure, they factorize into products of independent Bernoulli distributions.
- **Distribution of hidden units given visible units** $p(h|v)$:
$$ p(h|v) = \frac{p(v, h)}{p(v)} = \frac{\exp(-E(v, h))}{\sum_{h'} \exp(-E(v, h'))}$$
Because there are no hidden-hidden connections in an RBM, this distribution factorizes as 
$$p(\mathbf{h} \mid \mathbf{v}) = \prod_{j=1}^{N_h} p(h_j \mid \mathbf{v}),$$
where $p(h_j = 1 \mid \mathbf{v}) = \sigma\left( c_j + \sum_{i} w_{ij} v_i \right)$
, enabling parallel sampling.
- **Distribution of visible units given hidden units** $p(v|h)$: Similarly, we have
$$p(\mathbf{v} \mid \mathbf{h}) = \prod_{i=1}^{N_v} p(v_i \mid \mathbf{h})$$
where $p(v_i = 1 \mid \mathbf{h}) = \sigma\left( b_i + \sum_{j} w_{ij} h_j \right)$

## The Key Computational Advantage: Conditional Independence
The absence of intra-layer connections yields a crucial property: **conditional independence**. Given the visible units, the hidden units become independent of each other. Conversely, given the hidden units, the visible units become independent.

Mathematically, the conditional probability of a hidden unit $h_j$ being active given the visible layer is:

$$P(h_j = 1 \mid \mathbf{v}) = \sigma\left( c_j + \sum_{i=1}^{N_v} w_{ij} v_i \right)$$

where \( \sigma(z) = 1/(1 + \exp(-z)) \) is the sigmoid function. Because there are no hidden-hidden connections, the joint conditional distribution over all hidden units factorizes:

$$P(\mathbf{h} \mid \mathbf{v}) = \prod_{j=1}^{N_h} P(h_j \mid \mathbf{v})$$

Similarly, the conditional probability of a visible unit being active given the hidden layer is:

$$P(v_i = 1 \mid \mathbf{h}) = \sigma\left( b_i + \sum_{j=1}^{N_h} w_{ij} h_j \right)$$

and the visible units are conditionally independent given the hidden units:

$$P(\mathbf{v} \mid \mathbf{h}) = \prod_{i=1}^{N_v} P(v_i \mid \mathbf{h})$$

## Efficient Gibbs Sampling: Block Updates
The conditional independence property transforms Gibbs sampling from a slow, unit-by-unit process into a fast, **block-wise** process. A single round of Gibbs sampling consists of two parallel steps:

1. **Sample hidden layer given visible layer**: For each hidden unit $j$, compute $P(h_j = 1 \mid \mathbf{v})$ and sample $h_j \in \{0,1\}$ accordingly. All hidden units can be sampled simultaneously.
2. **Sample visible layer given hidden layer**: For each visible unit $i$, compute $P(v_i = 1 \mid \mathbf{h})$ and sample $ v_i \in \{0,1\}$. All visible units can be sampled simultaneously.

This pair of steps constitutes one full step of **block Gibbs sampling**. Starting from a visible vector $\mathbf{v}^{(0)}$, the sequence is:

$$\mathbf{v}^{(0)} \xrightarrow{P(\mathbf{h} \mid \mathbf{v}^{(0)})} \mathbf{h}^{(0)} \xrightarrow{P(\mathbf{v} \mid \mathbf{h}^{(0)})} \mathbf{v}^{(1)} \xrightarrow{P(\mathbf{h} \mid \mathbf{v}^{(1)})} \mathbf{h}^{(1)} \rightarrow \cdots$$

Each half-step updates an entire layer in parallel, making the RBM vastly more efficient than the classic Boltzmann machine, where each unit update requires computing inputs from all other units.

## Learning the RBM: Contrastive Divergence Revisited
The RBM is trained using the **contrastive divergence (CD)** algorithm introduced in Section [3.3 Contrastive Divergence](kpp-theoretical-foundations-ebms-cd.md). The gradient of the log-likelihood retains the same contrastive form:

$$\frac{\partial \log P(\mathbf{v})}{\partial w_{ij}} = \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}}$$

Thanks to the conditional independence property of the RBM, the Gibbs sampling steps required by CD become **block-parallel** operations. A single round-trip consists of sampling all hidden units given the visible layer, then sampling all visible units given the hidden layer. This makes even CD-1 remarkably effective. The weight update for a mini-batch is:

$$\Delta \mathbf{W} = \eta \left( \mathbb{E}[\mathbf{v} \mathbf{h}^\top]_{\text{data}} - \mathbb{E}[\mathbf{v} \mathbf{h}^\top]_{\text{recon}} \right)$$

with analogous updates for biases. For a full exposition of the CD-k algorithm, refer back to Section [3.3 Contrastive Divergence](kpp-theoretical-foundations-ebms-cd.md).

In practice, **CD-1**, i.e. k = 1, is often sufficient. The algorithm then becomes remarkably simple: for each data vector, compute hidden probabilities, sample hidden states, reconstruct visible probabilities, sample visible states, recompute hidden probabilities, and update weights based on the difference between the initial and reconstructed pairwise products.

## Free Energy and Marginal Probability
For an RBM, the free energy of a visible vector $\mathbf{v}$ is defined as:

$$F_\theta(\mathbf{v}) = -\log \sum_{\mathbf{h}} \exp\left(-E_\theta(\mathbf{v}, \mathbf{h})\right)$$

The marginal probability of a visible vector is then:

$$P_\theta(\mathbf{v}) = \frac{\exp(-F_\theta(\mathbf{v}))}{\sum_{\tilde{\mathbf{v}}} \exp(-F_\theta(\tilde{\mathbf{v}}))}$$

While the denominator (partition function over visible states) remains intractable, i.e. still sums over all $2^{N_v}$ visible configurations, the free energy itself is cheap to compute for any given $\mathbf{v}$. This allows, for example, comparing the relative likelihood of different visible vectors, even though the absolute likelihood remains intractable.

## Free Energy and Its Analytical Form
In an RBM, the hidden units are conditionally independent given the visible units. This allows us to sum out the hidden units analytically. Substituting the RBM energy function:

$$E_\theta(\mathbf{v}, \mathbf{h}) = -\mathbf{b}^\top \mathbf{v} - \mathbf{c}^\top \mathbf{h} - \mathbf{v}^\top \mathbf{W} \mathbf{h}$$

We can sum out the hidden units analytically:

$$\sum_{\mathbf{h}} \exp(-E_\theta(\mathbf{v}, \mathbf{h})) = \exp(\mathbf{b}^\top \mathbf{v}) \prod_{j=1}^{N_h} \left( 1 + \exp(c_j + \sum_i v_i w_{ij}) \right)$$

Thus, free energy has a closed-form expression:

$$F_\theta(\mathbf{v}) = -\mathbf{b}^\top \mathbf{v} - \sum_{j=1}^{N_h} \log \left( 1 + \exp\left( c_j + \sum_i v_i w_{ij} \right) \right)$$

**This analytical form is a key computational advantage of the RBM over the classic Boltzmann machine.** It allows us to evaluate the relative likelihood of different visible vectors in $O(N_v N_h)$ **time**, without enumerating the $2^{N_h}$ hidden configurations. However, **the absolute likelihood remains intractable**.

## Free Energy vs. Expected Energy

To gain deeper physical intuition, it is instructive to compare the free energy with the **expected energy** under the conditional distribution of hidden units:

$$\mathbb{E}_\theta[E_\theta(\mathbf{v}, \mathbf{H}) \mid \mathbf{v}] = -\mathbf{b}^\top \mathbf{v} - \mathbf{c}^\top \mathbf{m}(\mathbf{v}) - \mathbf{v}^\top \mathbf{W} \mathbf{m}(\mathbf{v})$$

where $m_j(\mathbf{v}) = \sigma(c_j + \sum_i v_i w_{ij})$ is the conditional expectation of hidden unit $j$ given $\mathbf{v}$. The free energy and expected energy are related by:

$$F_\theta(\mathbf{v}) = \mathbb{E}_\theta[E_\theta(\mathbf{v}, \mathbf{H}) \mid \mathbf{v}] - \text{Entropy}(P_\theta(\mathbf{H} \mid \mathbf{v}))$$

The difference is precisely the **entropy** of the hidden layer given the visible layer. This aligns with the thermodynamic relation $F = U - TS$: free energy equals the expected energy minus the temperature-weighted entropy. In the RBM, this entropy term accounts for uncertainty in the hidden unit activations.

## RBMs as Product of Experts
An RBM can be interpreted as a **product of experts** (PoE). Each hidden unit $h_j$ acts as an "expert" that assigns a probability to a visible vector $\mathbf{v}$ based on how well $\mathbf{v}$ matches its preferred pattern (encoded in the weight vector $\mathbf{w}_{:j}$). The joint distribution is the product of these expert opinions, renormalized:

$$P(\mathbf{v}) \propto \prod_{j=1}^{N_h} \exp\left( c_j h_j + \sum_i v_i w_{ij} h_j \right) \bigg|_{h_j \text{ summed out}}$$

The product-of-experts view highlights the RBM's representational power: each hidden unit contributes a soft constraint, and the combination of many such constraints can model complex, high-dimensional distributions. The bipartite restriction prevents experts from directly interacting, forcing them to coordinate only through the visible layer.

## Variants: Handling Different Data Types
The basic RBM with binary visible units is suitable for modeling binary data (e.g., black-and-white images, binary feature vectors). However, the RBM framework extends naturally to other data types by modifying the energy function and the conditional distributions.

- **Gaussian-Bernoulli RBM**: For real-valued data (e.g., grayscale images, continuous features), the visible units are modeled as Gaussian with unit variance (or learned variance). The energy becomes:

$$E(\mathbf{v}, \mathbf{h}) = \sum_i \frac{(v_i - b_i)^2}{2\sigma_i^2} - \sum_j c_j h_j - \sum_{i,j} \frac{v_i}{\sigma_i} w_{ij} h_j$$
The conditional $P(v_i \mid \mathbf{h})$ is Gaussian with mean $b_i + \sigma_i \sum_{j} w_{ij} h_j$

- **Softmax / Multinomial RBM**: For count data or categorical data (e.g., word counts, user ratings), the visible layer can be a softmax unit over a vocabulary or rating scale.
- **Replicated Softmax RBM**: For variable-length documents, the visible layer consists of a set of multinomial units, modeling the distribution of word counts.

These variants preserve the bipartite structure and conditional independence properties, allowing the same efficient block Gibbs sampling and contrastive divergence training.

## Limitations of the RBM
Despite its success, the RBM has limitations:

1. **Shallow Architecture**: A single RBM is a shallow model with only one layer of latent variables. Its representational power is limited compared to deep architectures, although stacking RBMs to form a **Deep Belief Network (DBN)** mitigates this (see Section [4.3 Beyond Single Layers: Stacking for Deep Learning](kpp-theoretical-foundations-bm-deep.md)).
2. **Partition Function Still Intractable**: The marginal likelihood $P(\mathbf{v})$ still requires summing over all visible configurations. While inference is efficient, model evaluation and comparison remain challenging.
3. **Mode Collapse in CD Training**: As noted in Section [3.3 Contrastive Divergence](kpp-theoretical-foundations-ebms-cd.md), CD training can suffer from mode collapse, where the model fails to capture all modes of the data distribution. Persistent CD (PCD) improves this but adds computational overhead.
4. **Hyperparameter Sensitivity**: The quality of the learned features depends sensitively on the learning rate, momentum, weight decay, and the number of hidden units. Tuning these hyperparameters requires experience and careful validation.

## The RBM's Place in Deep Learning History
The RBM played a pivotal role in the **deep learning renaissance** of the mid-2000s. Before the widespread adoption of rectified linear units (ReLUs), batch normalization, and improved optimizers for feedforward networks, training deep neural networks was notoriously difficult. The RBM provided an effective **unsupervised pre-training** method: by stacking RBMs trained layer by layer, one could initialize a deep network with sensible features extracted from the data distribution, rather than random weights. This pre-training, followed by discriminative fine-tuning with backpropagation, enabled the first successful training of truly deep architectures on challenging tasks like MNIST and later ImageNet.

Although modern deep learning has largely moved away from RBM pre-training in favor of end-to-end supervised training with large labeled datasets, the conceptual contributions of the RBM, i.e. energy-based learning, contrastive divergence, and the power of distributed representations, remain foundational. Moreover, RBMs continue to find niche applications in collaborative filtering, dimensionality reduction, and generative modeling of small-to-medium datasets.

## Summary
- The **Restricted Boltzmann Machine (RBM)** imposes a **bipartite connectivity** constraint: no visible-visible or hidden-hidden connections.
- This restriction yields **conditional independence**: $P(\mathbf{h} \mid \mathbf{v})$ and $P(\mathbf{v} \mid \mathbf{h})$ factorize into products of independent Bernoulli distributions.
- **Block Gibbs sampling** updates entire layers in parallel, dramatically accelerating inference and training.
- The RBM is trained using **contrastive divergence (CD)**, typically CD-1, which requires only one round-trip of block Gibbs sampling per data point.
- The **free energy** of a visible vector has a closed-form expression, enabling efficient comparison of relative likelihoods.
- The RBM can be extended to handle **real-valued** (Gaussian-Bernoulli RBM) and **categorical** (Softmax RBM) data.
- RBMs served as the building blocks for **Deep Belief Networks** and played a historic role in enabling unsupervised pre-training for deep neural networks.
- While no longer the dominant paradigm for supervised deep learning, RBMs remain important models for unsupervised learning, feature extraction, and understanding energy-based generative models.
