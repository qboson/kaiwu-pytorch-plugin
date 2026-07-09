---
title: 3.3 Contrastive Divergence
slug: kpp-theoretical-foundations-ebms-cd
sidebar_position: 9
hide: false
hide_child: false
---


# 3.3 Contrastive Divergence

> Section [3.2 The Intractable Partition Function Problem](kpp-theoretical-foundations-ebms-partition.md) established that exact maximum likelihood learning of Boltzmann machines is computationally infeasible due to the intractable partition function. The model expectation term in the gradient requires sampling from the equilibrium distribution $P_\theta$, which in turn demands running a Markov chain to convergence, which is a prohibitively slow process. In this section, we introduce **contrastive divergence (CD)**, the algorithmic breakthrough that made Boltzmann machines practical for real-world applications. CD provides a simple, efficient approximation to the gradient that circumvents the need for equilibrium sampling, trading theoretical purity for empirical effectiveness.

### The Core Insight: Truncated Markov Chains
The key observation behind contrastive divergence, introduced by Geoffrey Hinton in 2002, is that we need not run the Markov chain all the way to equilibrium. Instead, we can initialize the chain at a **data point** and run it for only a small number of steps, often just **one,** and then treat the resulting state as an approximate sample from the model distribution.

Why should this work? Consider the gradient of the negative log-likelihood:

$$\nabla f(\theta) = -\mathbb{E}_{\text{data}} \left[ \frac{\partial E_\theta}{\partial \theta} \right] + \mathbb{E}_{\text{model}} \left[ \frac{\partial E_\theta}{\partial \theta} \right]$$

The first term is an expectation over the data distribution, which is easy to compute. But the second term requires sampling from the model distribution. 

In contrastive divergence, we replace the model expectation with an expectation over a distribution obtained after $k$ steps of Gibbs sampling initialized from the data, yielding the **CD-k** update rule:

$$\Delta \theta \propto -\mathbb{E}_{\text{data}} \left[ \frac{\partial E_\theta}{\partial \theta} \right] + \mathbb{E}_{\text{recon after } k \text{ steps}} \left[ \frac{\partial E_\theta}{\partial \theta} \right]$$

> Note that this is equivalent to the standard CD update $\Delta w_{ij} = \eta (\langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{recon}})$ because $\frac{\partial E_\theta}{\partial w_{ij}} = -v_i h_j$.

The intuition is that the Markov chain, even after a few steps, moves away from the data distribution toward the model distribution. The difference between the data statistics and the statistics after $k$ steps approximates the true gradient direction. Remarkably, for $k = 1$, this simple heuristic works well enough to train useful generative models.

### The CD-k Algorithm in Detail
The CD-k algorithm for training a Restricted Boltzmann Machine (RBM) or a general Boltzmann machine proceeds as follows: **Algorithm: Contrastive Divergence (CD-k)** For each training example $\mathbf{v}^{(0)}$ (the visible state):

1. **Positive Phase**: 
 - Clamp the visible units to the data vector $\mathbf{v}^{(0)}$. 
 - Compute the hidden unit activations $P(\mathbf{h} \mid \mathbf{v}^{(0)})$ and sample a binary hidden state $\mathbf{h}^{(0)}$. 
 - Record the outer product $\mathbf{v}^{(0)} (\mathbf{h}^{(0)})^\top$ as the **positive statistics**.

2. **Negative Phase** (Gibbs sampling for $k$ steps): 
 - For $t = 1$ to $k$: 
 - Sample visible reconstruction $ \mathbf{v}^{(t)} \sim P(\mathbf{v} \mid \mathbf{h}^{(t-1)})$.
 - Sample hidden reconstruction $\mathbf{h}^{(t)} \sim P(\mathbf{h} \mid \mathbf{v}^{(t)})$. 
 - Record the outer product $\mathbf{v}^{(k)} (\mathbf{h}^{(k)})^\top$ as the **negative statistics**.

3. **Weight Update**: 

Update each weight $w_{ij}$ using the difference between positive and negative statistics:

$$\Delta w_{ij} = \eta \left( \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{recon}} \right)$$

where $\eta$ is the learning rate. Biases are updated similarly using unit activations.

For **CD-1** (the most common variant), the negative phase consists of a single round-trip: 

$\mathbf{v}^{(0)} \rightarrow \mathbf{h}^{(0)} \rightarrow \mathbf{v}^{(1)} \rightarrow \mathbf{h}^{(1)}$.

### Geometric Interpretation: Approximating the Gradient
The contrastive divergence update can be understood geometrically. The true gradient of the log-likelihood points in the direction that reduces the energy of data configurations and increases the energy of configurations that the model currently deems probable. CD approximates this by comparing the data configurations to nearby **reconstructions**, which are configurations obtained by following the Markov chain a short distance.

Consider the energy landscape. The data points sit in (or near) basins of low energy. Running Gibbs sampling for a few steps moves the state slightly away from the data, typically toward regions of higher energy if the model has not yet fully learned the data distribution. The CD update pushes the energy **down** at the data point and **up** at the nearby reconstruction. This effectively deepens the basin around the data while raising the energy of the surrounding region, creating a steeper attractor.

Crucially, CD does **not** attempt to raise the energy of *all* other configurations equally. It focuses only on those configurations that are easily reachable from the data via short MCMC trajectories. This local contrast is sufficient to learn a good model in practice, even though it introduces a bias relative to the true maximum likelihood gradient.

### Why Contrastive Divergence Works: A Heuristic Justification
The empirical success of CD, especially CD-1, initially surprised many researchers. Several lines of reasoning help explain its effectiveness:

1. **Initialization at Data**: By starting the Markov chain at data points, we are sampling from regions of high data density. If the model distribution is already close to the data distribution, the chain will mix quickly, and a few steps suffice. If the model is far from the data, the chain moves away rapidly, providing a strong contrast signal.
2. **Gradient of the KL Divergence**: It can be shown that CD-k minimizes a different objective function: the difference between two Kullback-Leibler divergences:

$$\text{CD}_k = \text{KL}(P_{\text{data}} \| P_\theta) - \text{KL}(P_\theta^{(k)} \| P_\theta)$$

where $P_\theta^{(k)}$ is the distribution after $k$ steps of MCMC starting from the data. Minimizing this **contrastive divergence** encourages the model distribution to be close to the data distribution while simultaneously being far from the $k$ step distribution. This formulation justifies the name and provides a more principled foundation than a mere heuristic.

1. **Persistent Contrastive Divergence (PCD)**: An important variant, **persistent contrastive divergence**, maintains a persistent set of Markov chains whose states are retained across weight updates. Instead of reinitializing from data each time, the chains continue to evolve. PCD yields samples that are closer to the true model distribution and can improve the quality of the learned model, especially for deeper architectures.

### Limitations and Caveats
Contrastive divergence is a pragmatic approximation, not an exact algorithm. It has several known limitations:

1. **Bias**: CD does not produce an unbiased estimate of the true log-likelihood gradient. The bias can be large for small $k$, and the resulting parameter estimates are not maximum likelihood estimates. The learned model may not assign the highest possible likelihood to the data.
2. **Not a Proper Gradient of Any Function**: For $k > 1$, the CD update is not the gradient of any fixed objective function. This makes theoretical analysis difficult and complicates convergence monitoring.
3. **Mixing and Mode Collapse**: If the model distribution has widely separated modes, short MCMC runs starting from data may fail to explore the full landscape, leading to mode collapse—the model learns only a subset of the data distribution.
4. **Dependence on k**: Larger values of $k$ reduce bias but increase computational cost. In practice, CD-1 or CD-5 are common compromises.

Despite these limitations, contrastive divergence remains a cornerstone algorithm for training restricted Boltzmann machines and deep belief networks. Its simplicity, speed, and empirical effectiveness enabled the mid-2000s renaissance of deep learning based on layer-wise pretraining.

### From Contrastive Divergence to Modern Approximations
The success of CD inspired a family of related algorithms that improve upon its limitations:

- **Persistent Contrastive Divergence (PCD)**: Uses persistent Markov chains to obtain better negative samples.
- **Fast Persistent Contrastive Divergence (FPCD)**: Introduces additional fast-learning weights to improve mixing.
- **Parallel Tempering**: Runs multiple chains at different temperatures to overcome energy barriers.
- **Score Matching and Noise Contrastive Estimation**: Bypass the need for MCMC entirely, as mentioned in Section [3.2 The Intractable Partition Function Problem](kpp-theoretical-foundations-ebms-partition.md).

These methods represent the ongoing evolution of techniques for training energy-based models, all stemming from the fundamental insight that we can trade exactness for tractability in learning.

### Practical Implementation: Mini-Batches and Sampling
In practice, the expectation $\mathbb{E}_{\text{data}}$ is not computed over the full dataset, but over a **mini-batch** of $M$ samples. This reduces computational cost and provides a stochastic gradient that can often escape local minima. Typical mini-batch sizes range from 10 to 100.

For the Gibbs sampling in the negative phase, you will usually run the chain for $k$ steps (often $k=1$) starting from the data. This is the Contrastive Divergence-$k$ (CD-$k$) algorithm. While not strictly following the gradient of the log-likelihood, it works well in practice. When implementing the update, the weight change for a mini-batch becomes:

$$\Delta \mathbf{W} = \frac{\eta}{M} \sum_{m=1}^M \left( \langle \mathbf{v}^{(m)} (\mathbf{h}^{(m)})^\top \rangle_{\text{data}} - \langle \mathbf{v}^{(m)}_k (\mathbf{h}^{(m)}_k)^\top \rangle_{\text{recon}} \right)$$

where $\mathbf{v}^{(m)}_k, \mathbf{h}^{(m)}_k$ are the final states after $k$ Gibbs steps. Using mini-batches is crucial for scaling to large datasets like MNIST or ImageNet. **Efficiency tip**: In code (especially Python), try to vectorize the computation over the entire mini-batch using matrix operations rather than looping over individual examples. This can lead to orders-of-magnitude speedups.

### Summary
- **Contrastive divergence (CD)** is a practical approximation to the maximum likelihood gradient for training Boltzmann machines.
- CD replaces the model expectation with an expectation over a **short Markov chain** initialized at the data, typically running for only $k$ steps (often $k=1$).
- The CD-k algorithm alternates between a **positive phase** (clamped to data) and a **negative phase** (brief free-running reconstruction).
- Geometrically, CD deepens the energy basin at data points while raising energy in their immediate MCMC-accessible neighborhood.
- CD minimizes a **difference of KL divergences** rather than the negative log-likelihood directly.
- Limitations include **bias**, lack of a proper objective function for $ k>1$, and potential **mode collapse**.
- **Persistent contrastive divergence (PCD)** improves sampling quality by maintaining persistent Markov chains across updates.
- **Mini-batch training** and **vectorization** are key practical considerations for efficient implementation.
- CD enabled the practical training of RBMs and deep belief networks, playing a pivotal role in the deep learning renaissance of the 2000s.
