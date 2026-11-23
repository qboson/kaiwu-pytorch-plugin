# KPP RBM Handwritten Digit Recognition

This directory provides an RBM-based feature-learning and classification
example on the scikit-learn Digits dataset (8x8 handwritten digits) using
`kaiwu-pytorch-plugin`. A Restricted Boltzmann Machine (RBM) is trained in an
unsupervised way; its hidden-layer representations are then used as features
for logistic regression, and are compared against logistic regression trained
on raw pixels.

## Dependencies

```bash
# Core dependencies (run from the repository root)
pip install -r requirements/requirements.txt   # kaiwu==1.3.1, torch==2.7.0, numpy==2.2.6
pip install .                                  # install the plugin

# Example-side dependencies (scikit-learn, scipy, matplotlib, seaborn)
pip install -r requirements/requirements_example.txt
```

## File Structure

```text
rbm_digits/
├── rbm_digits.py            # RBMRunner: data loading/augmentation, RBM
│                            #   training, feature extraction, visualization
├── rbm_classifier.py        # train_classifier(): RBM features + logistic
│                            #   regression vs raw-pixel baseline
└── rbm_digits.ipynb         # Walkthrough: theory recap, training, evaluation
```

## Data

The example uses `sklearn.datasets.load_digits` (8x8 images, 10 classes).
Each image is expanded by shifting it up / down / left / right (5x total
samples), normalized with `MinMaxScaler`, and split into 80% train / 20%
test (`random_state=42`).

## Training

Run from inside `example/rbm_digits/` (the notebook and scripts import the
local modules):

- **Notebook**: open `rbm_digits.ipynb` and run the cells. The training cell
  calls `train_classifier(n_iter=5, use_cim=False)`; the final cells plot the
  learned weight matrix and the confusion matrix.
- **Scripts**: `rbm_classifier.py` exposes `train_classifier(n_iter, use_cim)`,
  which builds a `Pipeline([("rbm", RBMRunner), ("logistic", ...)])`, trains it,
  and also trains a raw-pixel logistic regression baseline.

`use_cim=True` switches the sampler from `SimulatedAnnealingOptimizer` to a
`CIMOptimizer` (+ `PrecisionReducer`) and requires real-machine access — see
the repository-level README for obtaining a QPU quota.

## Outputs

- **Console**: test accuracy and `classification_report` for both the
  RBM-feature model and the raw-pixel baseline.
- **Plots** (displayed in the notebook; saved as PDF under `results/` when
  `save_pdf=True`):
  - `qbm_weights.pdf` — learned RBM weight matrix
  - `qbm_reconstructed_images.pdf` — original vs reconstructed images
  - `rbm_confusion_matrix_<suffix>.pdf` — confusion matrix

## Evaluation

Classification quality is measured by test accuracy and the per-class
`classification_report` of logistic regression on RBM hidden features versus
logistic regression on raw pixels.

## Key Parameters

`RBMRunner` (defaults):

```text
n_components     Number of hidden units (default: 256; classifier demo: 128)
learning_rate    SGD learning rate for the RBM (default: 0.1)
batch_size       Batch size (default: 100; classifier demo: 32)
n_iter           Number of training iterations (default: 30; notebook: 5-6)
use_cim          Use the CIM quantum sampler (requires QPU quota)
plot_img         Visualize generated samples and weights during training
random_state     Random seed
```

`train_classifier` / logistic baseline: `LogisticRegression(C=500.0,
max_iter=1000)`; sampler: `SimulatedAnnealingOptimizer(alpha=0.999,
size_limit=100)`.
