# RBM Classification: Handwritten Digit Recognition

> This tutorial builds on the [Quick Start](../quickstart.md) and [Section 4.2: Restricted Boltzmann Machine](../../theoretical-foundations/kpp-theoretical-foundations-bm-restricted.md). It walks through a complete two-stage pipeline: train an RBM on handwritten digits, use its hidden activations as features, and attach a logistic regression classifier on top. Using the same positive/negative phase loop as the [Quick Start](../quickstart.md), this tutorial focuses on the feature-extraction pipeline and downstream classification.

## Overview

By the end of this tutorial you will be able to:

- Use an RBM as an unsupervised feature extractor on image data
- Compare RBM features against raw pixels in a downstream classification task
- Visualize learned RBM weights and evaluate the classifier's performance
- Visualize the confusion matrix of the trained classifier

## Running the Example

**Dependencies**:

```{code-block} bash
pip install scikit-learn matplotlib scipy
```

**Run**:

```bash
cd example/rbm_digits
jupyter notebook rbm_digits.ipynb
```

**Key files**:

- `rbm_digits.py`: `RBMRunner` — data loading, RBM training, weight and confusion-matrix visualization
- `rbm_classifier.py`: `train_classifier` — two-stage RBM + logistic regression pipeline
- `rbm_digits.ipynb`: interactive walkthrough

## How It Works

The RBM training loop is identical to the [Quick Start](../quickstart.md): positive phase from data, negative phase from `rbm.sample(sampler)`, and `rbm.objective(...)` provides the surrogate loss. The theoretical details of RBM energy functions and conditional independence are covered in [Section 4.2](../../theoretical-foundations/kpp-theoretical-foundations-bm-restricted.md); here we focus on why RBMs work as unsupervised feature extractors.

RBMs are bipartite energy-based models: visible and hidden layers are connected across, but not within, each layer. Lower-energy configurations have higher probability, and the hidden units learn to combine raw pixels into more abstract features — exactly what makes RBMs useful as a feature extraction stage before a downstream classifier.

The overall pipeline is:

```
raw pixels (64-d)  →  RBM forward  →  hidden features (n_hidden-d)  →  LogisticRegression  →  class
```

The dataset uses scikit-learn's 8×8 handwritten digits, augmented by shifting each image up, down, left, and right to expand the training set. The notebook compares the two-stage RBM pipeline against a logistic regression on raw pixels to demonstrate that unsupervised RBM pretraining improves classification accuracy.

## Quick Start

The following snippet shows how to run the pipeline from a user's perspective, this is exactly what you would execute in a notebook:

Load the digits dataset (with four-direction shift augmentation):

```python
from rbm_digits import RBMRunner

runner = RBMRunner(n_hidden=64, device="cpu")
X_train, X_test, y_train, y_test = runner.load_data()
```

Train the RBM:

```python
runner.fit(X_train, epochs=20, batch_size=64, lr=0.1, verbose=True)
```

Build and evaluate the two-stage classifier (RBM features + logistic regression) versus raw-pixel baseline:

```python
from rbm_classifier import train_classifier

accuracy_rbm, accuracy_raw = train_classifier(runner, X_train, X_test, y_train, y_test)
print(f"RBM features: {accuracy_rbm:.4f} | Raw pixels: {accuracy_raw:.4f}")
```

## Key Code

### 1. Sampler Initialization (SA vs CIM)

The `RBMRunner` class supports switching between classical Simulated Annealing (SA) and quantum Coherent Ising Machine (CIM) samplers:

```{literalinclude} ../../../../example/rbm_digits/rbm_digits.py
:pyobject: RBMRunner.__init__
```

When `use_cim=True`:
- Uses `CIMOptimizer` from `kaiwu.cim`
- Wrapped in `PrecisionReducer` for precision control
- Checkpoint manager saves to `./tmp`

When `use_cim=False`:
- Uses `SimulatedAnnealingOptimizer` from `kaiwu.classical`

---

### 2. Training Loop

The training loop follows the same positive/negative phase pattern introduced in the [Quick Start](../quickstart.md), see the training flow diagram there.

```{literalinclude} ../../../../example/rbm_digits/rbm_digits.py
:pyobject: RBMRunner.fit
```

Key training details:
- Optimizer: SGD with learning rate from `self.learning_rate`
- Weight decay: 0.02 for weights, 0.05 for biases
- Verbose mode prints objective loss and weight statistics every 20 iterations

---

### 3. Feature Extraction and Classification

After training, `RBMRunner.transform()` extracts hidden-layer features, which are then fed to a downstream classifier:

```{literalinclude} ../../../../example/rbm_digits/rbm_classifier.py
:pyobject: train_classifier
```

## Training and Evaluation

- **Weight visualization**: `RBMRunner.plot_weights()` renders each hidden unit's filter as an 8×8 image, showing the digit-like features the RBM has learned.
- **Confusion matrix**: `RBMRunner.plot_confusion_matrix(y_test, y_pred)` produces a heatmap; the notebook saves it as a high-resolution PDF.
- The expected result is that RBM features yield a 2–5 percentage point accuracy gain over raw pixels on the 8×8 digits dataset.

## Summary

- An RBM can be used as a fixed feature extractor after unsupervised pretraining.
- The KPP training loop (positive / negative phase / objective) is unchanged from the Quick Start.
- The downstream classifier is a standard scikit-learn pipeline, KPP only contributes the learned features.
