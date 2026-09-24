# Prerequisites

Restricted Boltzmann Machine (RBM) is an energy-based probabilistic graphical model, composed of a visible layer and a hidden layer, with no connections within each layer and full connections between layers. Its core objective is to learn the latent feature distribution of data through unsupervised learning.

---

## 1. Neural Network Basics

### 1.1 Neuron Model

An artificial neuron is the basic computational unit of a neural network. Given an input vector $\mathbf{x} \in \mathbb{R}^n$, its output is:

$$
a = \phi\left( \mathbf{w}^\top \mathbf{x} + b \right)
$$

where $\mathbf{w} \in \mathbb{R}^n$ is the weight vector, $b \in \mathbb{R}$ is the bias, and $\phi(\cdot)$ is the activation function. In probabilistic generative models, the Sigmoid activation function is commonly used:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

### 1.2 Energy-Based Models

Unlike feedforward networks, Energy-Based Models (EBMs) define a probability distribution over data via a scalar energy function $E(\mathbf{x}; \theta)$:

$$
P(\mathbf{x}; \theta) = \frac{\exp(-E(\mathbf{x}; \theta))}{Z(\theta)}
$$

where the partition function is:

$$
Z(\theta) = \sum_{\mathbf{x}} \exp(-E(\mathbf{x}; \theta))
$$

ensuring normalization. Low energy states correspond to high probability.

---

## 2. Boltzmann Machine Architecture

- **Visible layer (v)**: explicit representation of input data (e.g., pixel values).
- **Hidden layer (h)**: extracted latent features.
- **Weight matrix (W)**: connections between visible and hidden layers.
- **Biases**: visible bias ($\mathbf{b}$) and hidden bias ($\mathbf{c}$).

A Boltzmann Machine (BM) is fully connected, while a Restricted Boltzmann Machine (RBM) removes intra-layer connections, making Gibbs sampling more efficient.

Due to the restricted structure of RBM, hidden variables are mutually independent given the visible variables, and their conditional probabilities are:

$$
P(h_j = 1 \mid \mathbf{v}) = \sigma\left( \sum_i w_{ij} v_i + c_j \right)
$$

Similarly,

$$
P(v_i = 1 \mid \mathbf{h}) = \sigma\left( \sum_j w_{ij} h_j + b_i \right)
$$

---

## 3. Energy Function and Probability Distribution

### 3.1 Energy Function

The energy function of an RBM is defined as:

$$
E(\mathbf{v}, \mathbf{h}) = -\mathbf{v}^T \mathbf{W} \mathbf{h} - \mathbf{b}^T \mathbf{v} - \mathbf{c}^T \mathbf{h}
$$

where $\mathbf{v}, \mathbf{h}$ are the states of the visible and hidden layers, $\mathbf{W}$ is the connection weight, and $\mathbf{b}, \mathbf{c}$ are bias terms.

The joint probability distribution is given by the Boltzmann distribution:

$$
P(\mathbf{v}, \mathbf{h}) = \frac{e^{-E(\mathbf{v}, \mathbf{h})}}{Z}
$$

where $Z$ is the partition function (normalization factor). The marginal distribution over the visible layer is:

$$
P(\mathbf{v}) = \sum_{\mathbf{h}} P(\mathbf{v}, \mathbf{h})
$$

We learn parameters $W, b, c$ by maximizing the likelihood. The objective is the negative log-likelihood:

$$
\mathcal{L} = -\sum_{\mathbf{v}} \log P(\mathbf{v})
$$

The Contrastive Divergence (CD) algorithm approximates the gradient, giving the update rule:

$$
\Delta W_{ij} = \epsilon \left( \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{recon}} \right)
$$

where $\epsilon$ is the learning rate, $\langle \cdot \rangle_{\text{data}}$ and $\langle \cdot \rangle_{\text{recon}}$ are the expectations under the data distribution and the reconstruction distribution, respectively.

### 3.2 Derivation of the Gradient

For an energy-based model, the probability can be written as:

$$
p(x; \theta) = \frac{1}{Z} \tilde{p}(x; \theta)
$$

Its gradient is:

$$
\nabla_\theta \log p(x; \theta) = \nabla_\theta \log \tilde{p}(x; \theta) - \nabla_\theta \log Z
$$

The gradient of the partition function is not directly computable:

$$
\begin{aligned}
\nabla_\theta \log Z
&= \frac{\nabla_\theta Z}{Z} \\
&= \frac{\nabla_\theta \sum_x \tilde{p}(x)}{Z} \\
&= \sum_x \frac{\nabla_\theta \tilde{p}(x)}{Z}
\end{aligned}
$$

For models where $p(x) > 0$ for all $x$, we can replace $\tilde{p}(x)$ with $\exp(\log \tilde{p}(x))$:

$$
\begin{aligned}
\frac{\sum_x \nabla_\theta \exp(\log \tilde{p}(x))}{Z}
&= \frac{\sum_x \exp(\log \tilde{p}(x)) \nabla_\theta \log \tilde{p}(x)}{Z} \\
&= \frac{\sum_x \tilde{p}(x) \nabla_\theta \log \tilde{p}(x)}{Z} \\
&= \sum_x p(x) \nabla_\theta \log \tilde{p}(x) \\
&= \mathbb{E}_{x \sim p(x)} \nabla_\theta \log \tilde{p}(x)
\end{aligned}
$$

Therefore,

$$
\nabla_\theta \log p(x; \theta) = \nabla_\theta \log \hat{p}(x; \theta) - \mathbb{E}_{x \sim p(x; \theta)} \nabla_\theta \log \hat{p}(x; \theta)
$$

The second term involves the model distribution $p(x; \theta)$, while the first term is from the empirical data distribution. Thus,

$$
\nabla_\theta \log p(x; \theta) = \mathbb{E}_{x \sim p_{\text{data}}} \nabla_\theta \log \hat{p}(x; \theta) - \mathbb{E}_{x \sim p_{\text{model}}} \nabla_\theta \log \hat{p}(x; \theta)
$$

For the Boltzmann machine energy function, we easily obtain:

$$
\nabla_W \log \hat{p}(x; W) = v h^\mathrm{T}
$$

Hence, the gradient can be computed by sampling $v$ and $h$ under $p_{\text{data}}$ and $p_{\text{model}}$, leading to:

$$
\Delta W_{ij} = \epsilon \left( \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{recon}} \right)
$$

## Further Reading

For a comprehensive treatment of the theoretical foundations behind energy-based models, sampling, and quantum enhancement, refer to the [Theoretical Foundations](../theoretical-foundations/index.md) section of the documentation.