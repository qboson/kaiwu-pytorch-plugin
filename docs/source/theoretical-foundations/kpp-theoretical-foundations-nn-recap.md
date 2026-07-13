---
title: '2.1 A Brief Recap: Linear Neurons and Limitations'
slug: kpp-theoretical-foundations-nn-recap
sidebar_position: 4
hide: false
hide_child: false
---


# 2.1 A Brief Recap: Linear Neurons and Limitations

> The previous chapter situated the Boltzmann machine within the conceptual landscape of statistical physics—a landscape defined by energy functions, equilibrium distributions, and thermal fluctuations. Before we delve deeper into energy-based learning, it is essential to revisit the elementary building blocks of neural computation. Understanding the simplest neuron models, their capabilities, and—crucially—their limitations provides the necessary contrast to appreciate why the Boltzmann machine adopts such a radically different computational paradigm.

### The Linear Neuron: Weighted Sum and Threshold
The most basic computational unit in neural networks is the **linear neuron** (also called a *linear threshold unit* or, historically, a *McCulloch-Pitts neuron*). Its operation is straightforward: it computes a weighted sum of its inputs and produces an output based on whether that sum exceeds a threshold.

Let $\mathbf{x} = (x_1, x_2, \ldots, x_d)$ be a vector of input values. Each input $x_j$ is multiplied by a corresponding **weight** $w_j$, representing the strength of the synaptic connection. An additional **bias** term $b$ shifts the decision boundary. The neuron's output $y$ is given by:

$$y = \begin{cases}1 & \text{if } \sum_{j=1}^d w_j x_j + b \geq 0 \\0 & \text{otherwise}\end{cases}$$

Equivalently, we can write this as:

$$y = \mathbb{I}\left( \mathbf{w}^\top \mathbf{x} + b \geq 0 \right)$$

where $\mathbb{I}(\cdot)$ is the indicator function. The vector $ \mathbf{w} = (w_1, \ldots, w_d)$ is called the **weight vector**.

### Geometric Interpretation: Linear Separability
The linear neuron has a clear geometric interpretation. The equation $\mathbf{w}^\top \mathbf{x} + b = 0$ defines a **hyperplane** in the $d$-dimensional input space. This hyperplane partitions the space into two half-spaces:

- Points on one side produce an output of 1.
- Points on the other side produce an output of 0.

The weight vector $\mathbf{w}$ is perpendicular to the hyperplane and points toward the positive half-space. The bias $b$ determines the perpendicular distance from the origin to the hyperplane (specifically, distance $= -b / \|\mathbf{w}\|$).

This geometric view immediately reveals the fundamental capability and the fundamental limitation of linear neurons. A linear neuron can successfully classify any dataset that is **linearly separable,** which means there exists some hyperplane that perfectly separates the positive examples (label 1) from the negative examples (label 0). For such problems, there exist weight vectors $\mathbf{w}$ and biases $b$ that achieve perfect accuracy.

### The Perceptron Learning Algorithm
The **perceptron** is a specific training algorithm for linear threshold units, introduced by Frank Rosenblatt in 1958. It provides a simple, iterative procedure for finding a separating hyperplane, *provided one exists*.

The perceptron learning rule processes training examples one at a time. For each example $(\mathbf{x}, t)$, where $t \in \{0,1\}$ is the target label, the algorithm:

1. Computes the current output: $y = \mathbb{I}(\mathbf{w}^\top \mathbf{x} + b \geq 0)$.
2. If $y = t$, the weights remain unchanged.
3. If $y \neq t$, the weights are updated as follows:
 - If $t = 1$ but $y = 0$ (false negative): $\mathbf{w} \leftarrow \mathbf{w} + \eta \mathbf{x}$, $b \leftarrow b + \eta$.
 - If $t = 0$ but $y = 1$ (false positive): $\mathbf{w} \leftarrow \mathbf{w} - \eta \mathbf{x}$, $b \leftarrow b - \eta$.

Here, $\eta > 0$ is the **learning rate**, a small positive constant controlling the step size.

The **Perceptron Convergence Theorem** guarantees that if the training data are linearly separable, this algorithm will find a separating hyperplane in a finite number of steps. This theoretical guarantee, proven in the early 1960s, generated enormous enthusiasm for neural network research.

### The Limits of Linear Separability: The XOR Problem
The enthusiasm was short-lived. In 1969, Marvin Minsky and Seymour Papert published their influential book *Perceptrons*, which rigorously analyzed the capabilities and limitations of linear threshold units. Their most famous counterexample is the **XOR (exclusive-or)** problem.

Consider two binary inputs $x_1, x_2 \in \{0,1\}$ and the desired outputs:

|$$x_1$$|$$x_2$$|**XOR Output**|
|---|---|---|
|0|0|0|
|0|1|1|
|1|0|1|
|1|1|0|

Plotting these four points in the 2D input plane reveals the difficulty. The positive examples, (0,1) and (1,0), are diagonally opposite; the negative examples, (0,0) and (1,1), occupy the other diagonal. **No single straight line can separate the two classes.** The XOR function is not linearly separable.

Minsky and Papert proved that this limitation extends far beyond XOR. Linear threshold units fail on any function that requires a *non-convex* decision boundary or one with disconnected regions. Many seemingly simple logical functions, i.e. parity, connectedness in visual patterns are beyond the reach of a single perceptron.

### The Historical Impact: The First AI Winter
The rigorous demonstration of the perceptron's limitations had a chilling effect on neural network research throughout the 1970s. Funding dried up, and many researchers abandoned connectionist approaches in favor of symbolic artificial intelligence. This period is often referred to as the **first AI winter**.

The core insight from this episode, however, was not that neural networks were fundamentally flawed, but that **single-layer linear networks are insufficient for complex tasks**. The path forward required two essential innovations:

1. **Multiple Layers**: Stacking linear threshold units into layers, with the outputs of one layer serving as inputs to the next, can overcome the linear separability limitation. A two-layer network can represent any Boolean function; with continuous activation functions, deeper networks can approximate arbitrarily complex decision boundaries.
2. **Nonlinear Activation Functions**: The step function $\mathbb{I}(z \geq 0)$ is discontinuous and non-differentiable. Replacing it with smooth, differentiable nonlinearities—such as the **sigmoid** $\sigma(z) = 1/(1+e^{-z})$ or the **hyperbolic tangent** $\tanh(z)$, enables gradient-based learning through **backpropagation**, a topic we will revisit in the context of energy-based models.

### From Feedforward Classification to Probabilistic Generation
The perceptron and its multilayer successors are fundamentally **discriminative models**: they learn a direct mapping from inputs $\mathbf{x}$ to outputs $y$. Their objective is to minimize classification errors on a labeled dataset. The computation is **feedforward**: information flows in one direction, from input to output, without cycles or recurrence.

The Boltzmann machine, in stark contrast, belongs to the family of **generative models**. Its objective is not to map inputs to outputs but to model the *joint probability distribution* $P(\mathbf{x})$ (or $P(\mathbf{x}, \mathbf{h})$ with hidden units) of the observed data. The computation is **recurrent**: units are interconnected symmetrically, and the network's state evolves over time through stochastic updates until equilibrium is reached.

Why abandon the simplicity of feedforward classification for the complexity of recurrent, stochastic, energy-based models? The answer lies in the types of problems each paradigm addresses. Perceptrons excel at *supervised pattern recognition* when large labeled datasets are available. Boltzmann machines address *unsupervised learning* and *generation*: discovering the underlying structure of data without explicit labels, completing missing information, and synthesizing new, plausible examples.

The limitations of linear neurons taught us that simple, shallow architectures cannot capture the rich structure of natural data. The Boltzmann machine answers this lesson not by stacking deterministic layers, but by embracing a fundamentally different computational metaphor—one drawn from the statistical physics of complex systems, where global structure emerges from local, stochastic interactions.

### Connection to Energy-Based Models
As we transition to the next section, note a subtle but important parallel. The perceptron's decision rule $\mathbf{w}^\top \mathbf{x} + b \geq 0$ can be reinterpreted in energetic terms. Define an **energy** for a given input-output pair:

$$E(\mathbf{x}, y) = -y (\mathbf{w}^\top \mathbf{x} + b)$$

The perceptron outputs $y=1$ when this energy is negative (i.e., low), and $y=0$ when it is zero (i.e., higher). The learning rule adjusts weights to lower the energy of correct classifications and raises the energy of incorrect ones.

This energy-based perspective, while rudimentary in the perceptron, becomes the central organizing principle in Boltzmann machines. The difference is that Boltzmann machines define energy over *all* units, both visible and hidden and use stochastic, rather than deterministic, state transitions to explore the energy landscape. The following sections will develop this energy-based viewpoint in full mathematical detail.

### Summary
- A **linear neuron** computes a weighted sum of inputs and applies a threshold, partitioning the input space with a hyperplane.
- The **perceptron algorithm** provably finds a separating hyperplane for linearly separable data.
- The **XOR problem** demonstrates that single linear threshold units cannot solve non-linearly separable tasks, triggering the first AI winter.
- Overcoming this limitation requires either **multiple layers** of nonlinear units or a fundamentally different computational paradigm.
- The **Boltzmann machine** chooses the latter path: a **generative**, **recurrent**, and **energy-based** architecture capable of unsupervised learning and probabilistic inference.
- The transition from deterministic perceptrons to stochastic Boltzmann machines parallels a shift from **discriminative classification** to **probabilistic modeling** of data distributions.

