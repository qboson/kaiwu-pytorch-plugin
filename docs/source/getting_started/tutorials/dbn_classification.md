# DBN Classification: Stacked RBM Pretraining and Supervised Fine-tuning

> This tutorial builds on the [RBM Classification tutorial](rbm_classification.md) and [Section 4.2: Restricted Boltzmann Machine](../../theoretical-foundations/kpp-theoretical-foundations-bm-restricted.md). A Deep Belief Network (DBN) stacks multiple RBM layers: each RBM is pretrained greedily and unsupervised, then the whole network is fine-tuned with labeled data. The key new concept is layer-wise pretraining — the same RBM objective applied sequentially. This tutorial also covers three training modes: unsupervised feature extraction, downstream classifier, and end-to-end fine-tuning.

## Overview

By the end of this tutorial you will be able to:

- Understand how stacking RBMs forms a DBN
- Run greedy layer-wise unsupervised pretraining
- Choose between three training modes: unsupervised feature extraction, downstream classifier, or end-to-end fine-tuning
- Compare supervised DBN against a single RBM classifier

## Running the Example

**Dependencies**:

```{code-block} bash
pip install scikit-learn matplotlib scipy
```

**Run**:

```bash
cd example/dbn_digits
jupyter notebook supervised_dbn_digits.ipynb
```

**Key files**:

- `dbn_trainer.py`: `DBNPretrainer` — greedy layer-wise RBM pretraining (internally delegates each layer to a low-level `DBNTrainer`)
- `supervised_dbn_digits.py`: `SupervisedDBNClassification` — supervised fine-tuning and classification
- `supervised_dbn_digits.ipynb`: interactive walkthrough

## How It Works

A DBN is a stack of RBMs:

```
x (64-d)  →  RBM₁  →  h₁ (100-d)  →  RBM₂  →  h₂ (100-d)  →  classifier
```

**Pretraining** is done greedily, one RBM at a time:

1. Train RBM₁ on the input data $x$ (standard RBM objective).
2. Freeze RBM₁, transform $x$ to get $h_1$, and train RBM₂ on $h_1$.
3. Each layer uses the same `objective()` and `sample()` as the [Quick Start](../quickstart.md).

The codebase provides a layered class structure:

- **`DBNPretrainer`**: stacks multiple RBM layers and runs greedy unsupervised pretraining layer by layer (internally delegating each layer to a low-level `DBNTrainer`).
- **`AbstractSupervisedDBN`**: defines the general interface for supervised DBN training (pretraining, fine-tuning, classification, prediction).
- **`AbstractSupervisedDBNClassifier`**: extends the abstract interface with common classifiers (logistic regression, SVM, random forest).
- **`SupervisedDBNClassification`**: concrete implementation for classification tasks, including fine-tuning network construction.

### Three Training Modes

After pretraining, three training modes are available:

- **Unsupervised mode**: only pretrain, use the DBN as a fixed feature extractor (`DBNPretrainer.transform`).
- **Classifier mode**: attach a downstream classifier (logistic regression, SVM, or random forest) on top of the last layer, no backprop through the RBMs.
- **Fine-tuning mode**: unroll the stack into a deep neural network initialized with pretrained RBM weights, then backpropagate with labels using `CrossEntropyLoss` and SGD with L2 regularization. Dropout can be enabled to prevent overfitting.

The sampler configuration follows the [Quick Start](../quickstart.md) pattern, per RBM layer.

## Quick Start

The steps below condense `example/dbn_digits/supervised_dbn_digits.ipynb` with imports omitted; the notebook is the fully runnable demo.

**1. Load the data.** `load_data` wraps `sklearn.datasets.load_digits` and augments the 8×8 images by shifting them in all four directions.

```python
X_train, X_test, y_train, y_test = load_data(plot_img=False)
```

**2. Configure the DBN.** Layer-wise RBM pretraining happens inside `fit`; pass the pretraining config (layer sizes, per-layer RBM epochs, batch size) and the training mode directly as keyword arguments. `fine_tuning=True` unrolls the stack into a deep network and backpropagates with labels; `fine_tuning=False` attaches a downstream classifier selected by `classifier_type`.

```python
dbn = SupervisedDBNClassification(
    hidden_layers_structure=[128, 256],
    learning_rate_rbm=0.1,
    n_epochs_rbm=2,
    batch_size=64,
    fine_tuning=True,
    classifier_type="logistic",   # used when fine_tuning=False
    learning_rate=0.1,
    n_iter_backprop=100,
    l2_regularization=1e-4,
    activation_function="sigmoid",
    verbose=True,
    random_state=42,
)
```

**3. Train and evaluate.** `fit` runs pretraining and then the chosen mode; `score` reports test accuracy; `transform` extracts layer-wise features.

```python
dbn.fit(X_train, y_train)
accuracy = dbn.score(X_test, y_test)
features = dbn.transform(X_test)
```

## Key Code

### 1. DBNPretrainer Initialization (SA vs CIM per layer)

The `DBNPretrainer` class is the pretraining entry point that `SupervisedDBNClassification` instantiates internally; it supports switching between classical Simulated Annealing (SA) and quantum Coherent Ising Machine (CIM) samplers per RBM layer:

```{literalinclude} ../../../../example/dbn_digits/dbn_trainer.py
:pyobject: DBNPretrainer.__init__
```

Internally, `DBNPretrainer` builds an `UnsupervisedDBN` and delegates each layer's training to a `DBNTrainer`.

When `use_cim=True`:
- Uses `CIMOptimizer` from `kaiwu.cim`
- Wrapped in `PrecisionReducer` for precision control
- Checkpoint manager saves to `./tmp`

When `use_cim=False`:
- Uses `SimulatedAnnealingOptimizer` from `kaiwu.classical`

---

### 2. Greedy Layer-wise Pretraining

`DBNPretrainer.fit` creates the RBM stack from the input dimension and iterates through each layer, training one at a time through the internal `DBNTrainer`:

```{literalinclude} ../../../../example/dbn_digits/dbn_trainer.py
:pyobject: DBNPretrainer.fit
```

Key pretraining details:
- Each RBM layer uses the same `objective()` and `sample()` as the Quick Start
- `create_rbm_layer(X.shape[1])` builds the RBM stack from the input dimension
- After training a layer, the output becomes input for the next layer
- Verbose mode prints progress for each layer

---

### 3. Supervised Classification and Fine-tuning

The `SupervisedDBNClassification` class provides three training modes: unsupervised feature extraction, downstream classifier, and end-to-end fine-tuning.

```{literalinclude} ../../../../example/dbn_digits/supervised_dbn_digits.py
:pyobject: SupervisedDBNClassification
```

## Training and Evaluation

- **Pretraining progress**: each RBM layer reports its objective loss; `RBMVisualizer` renders the learned weight filters at each layer and monitors weight gradients during training.
- **Reconstruction visualization**: the notebook encodes test images through the trained RBM stack and plots the reconstruction error evolution.
- **Generated samples**: during pretraining, the notebook can display samples generated by each RBM layer.
- **Confusion matrix**: after classification, `RBMVisualizer` produces a confusion matrix heatmap.
- **Comparison**: the notebook compares DBN (2-layer) against single-RBM features and raw pixels.

## Summary

- A DBN stacks RBMs, pretrained greedily one layer at a time using the same RBM objective.
- Three training modes: unsupervised feature extraction, downstream classifier, or end-to-end fine-tuning.
- The fine-tuning mode unrolls the RBM stack into a deep network initialized with pretrained weights, trained with `CrossEntropyLoss`.
- The sampler (SA vs CIM) is configured per RBM layer and follows the Quick Start pattern.
