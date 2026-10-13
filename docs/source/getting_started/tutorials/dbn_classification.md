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

- `dbn_trainer.py`: `DBNPretrainer` — greedy layer-wise RBM pretraining
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

- **`DBNPretrainer`**: stacks multiple RBM layers and runs greedy unsupervised pretraining layer by layer.
- **`AbstractSupervisedDBN`**: defines the general interface for supervised DBN training (pretraining, fine-tuning, classification, prediction).
- **`AbstractSupervisedDBNClassifier`**: extends the abstract interface with common classifiers (logistic regression, SVM, random forest).
- **`SupervisedDBNClassification`**: concrete implementation for classification tasks, including fine-tuning network construction.

### Three Training Modes

After pretraining, three training modes are available:

- **Unsupervised mode**: only pretrain, use the DBN as a fixed feature extractor (`DBNPretrainer.transform`).
- **Classifier mode**: attach a downstream classifier (logistic regression, SVM, or random forest) on top of the last layer, no backprop through the RBMs.
- **Fine-tuning mode**: unroll the stack into a deep neural network initialized with pretrained RBM weights, then backpropagate with labels using `CrossEntropyLoss` and SGD with L2 regularization. Dropout can be enabled to prevent overfitting.

The sampler configuration is inherited per RBM layer from the [Quick Start](../quickstart.md).

## Quick Start

The following snippet shows how to run the pipeline from a user's perspective, this is what you would execute in `supervised_dbn_digits.ipynb`:

Load digits data (same shift augmentation as the RBM tutorial):

```python
from dbn_trainer import DBNTrainer
from supervised_dbn_digits import SupervisedDBNClassification
from sklearn.datasets import load_digits

X_train, X_test, y_train, y_test = load_digits(shift_augment=True)
```

Greedy layer-wise pretraining:

```python
pretrainer = DBNTrainer(hidden_layers_structure=[100, 100], n_epochs_rbm=10, lr=0.1)
pretrainer.fit(X_train)
```

Supervised classification (classifier mode):

```python
clf = SupervisedDBNClassification(pretrainer, classifier_type="logistic")
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(f"DBN classifier accuracy: {accuracy:.4f}")
```

Fine-tuning mode (optional):

```python
clf_ft = SupervisedDBNClassification(pretrainer, fine_tuning=True, n_iter_backprop=100)
clf_ft.fit(X_train, y_train)
```

## Key Code

### 1. DBNTrainer Initialization (SA vs CIM per layer)

The `DBNTrainer` class supports switching between classical Simulated Annealing (SA) and quantum Coherent Ising Machine (CIM) samplers per RBM layer:

```{literalinclude} ../../../../example/dbn_digits/dbn_trainer.py
:pyobject: DBNTrainer.__init__
```

When `use_cim=True`:
- Uses `CIMOptimizer` from `kaiwu.cim`
- Wrapped in `PrecisionReducer` for precision control
- Checkpoint manager saves to `./tmp`

When `use_cim=False`:
- Uses `SimulatedAnnealingOptimizer` from `kaiwu.classical`

---

### 2. Greedy Layer-wise Pretraining

The `train()` method iterates through each RBM layer, training one at a time:

```{literalinclude} ../../../../example/dbn_digits/dbn_trainer.py
:pyobject: DBNTrainer.train
```

Key pretraining details:
- Each RBM layer uses the same `objective()` and `sample()` as the Quick Start
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
