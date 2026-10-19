# KPP DBN Handwritten Digit Recognition

This directory provides a complete Deep Belief Network (DBN) implementation for handwritten digit recognition on the Digits dataset using `kaiwu-pytorch-plugin`. Multiple RBMs are stacked and pre-trained layer-wise (unsupervised), then the network is used for classification via two strategies: **fine-tuning** (end-to-end backpropagation through a network initialized with the pre-trained weights) and **classifier mode** (traditional ML classifiers trained on DBN-extracted features).

## File Structure

```text
dbn_digits/
├── supervised_dbn_digits.py    # Data loading/augmentation,
│                               #   AbstractSupervisedDBN + SupervisedDBNClassification
│                               #   (fine-tuning & classifier modes), RBMVisualizer
├── dbn_trainer.py              # DBNTrainer + DBNPretrainer: layer-wise
│                               #   unsupervised pre-training of stacked RBMs
└── supervised_dbn_digits.ipynb # Walkthrough: theory recap + both modes demo
```

## Data

Same pipeline as `rbm_digits`: `sklearn.datasets.load_digits` (8x8), four-direction shift augmentation (5x), `MinMaxScaler` normalization, 80% / 20% split (`random_state=42`).

## Training

Run from inside `example/dbn_digits/`. The notebook (`supervised_dbn_digits.ipynb`) demonstrates both modes through `demonstrate_both_modes()`:

1. **Fine-tuning mode** (`fine_tuning=True`): pre-train each RBM layer greedily with `DBNPretrainer`, initialize a feed-forward network with the pre-trained weights, then train it end-to-end with backpropagation (`CrossEntropyLoss`, SGD, optional dropout).
2. **Classifier mode** (`fine_tuning=False`): extract features from the pre-trained DBN and train a traditional classifier (`classifier_type`: `logistic`, `svm`, or `random_forest`).

The DBN pretraining uses `UnsupervisedDBN` from `kaiwu.torch_plugin.dbn` with a `SimulatedAnnealingOptimizer` (or `CIMOptimizer` when `use_cim=True`, requiring real-machine access).

## Outputs

- **Console**: per-layer pre-training stats, fine-tuning loss/accuracy, and final test accuracy per mode.
- **`results/`** (created by the notebook / visualizer):
  - `<prefix>_structure.json`, `<prefix>_summary.txt` — network structure (layers, hidden units, classes, mode)
  - weight / reconstruction / confusion-matrix figures (when `save_pdf=True`)
- **`data/`** (via `save_parameters(prefix)`):
  - `*_pretrain_layer{i}_weights.npy` / `*_bias.npy` — pre-trained RBM layers
  - `*_finetune_layer{i}_weights.npy` / `*_bias.npy` — fine-tuned layers
  - `*_classifier.pkl` — saved classifier (classifier mode only)

## Evaluation

Each mode reports test accuracy / `score()` on the held-out set; the notebook also supports per-layer activation inspection (`get_layer_activations`), feature importance (`get_feature_importance`, classifier mode), and reconstruction-error analysis (`RBMVisualizer.plot_reconstructed_images`).

## Key Parameters

`SupervisedDBNClassification` (defaults):

```text
hidden_layers_structure   Hidden units per RBM layer (default: [100, 100];
                          notebook demo: [128, 256])
learning_rate_rbm         Pre-training learning rate (default: 0.1)
n_epochs_rbm              Pre-training epochs per layer (default: 10;
                          notebook demo: 2)
batch_size                Batch size (default: 32; notebook demo: 64)
fine_tuning               True: end-to-end fine-tuning mode;
                          False: classifier mode
learning_rate             Fine-tuning learning rate (default: 0.1)
n_iter_backprop           Fine-tuning iterations (default: 100)
l2_regularization         L2 weight decay for fine-tuning (default: 1e-4)
activation_function       sigmoid or relu (default: sigmoid)
dropout_p                 Dropout probability (default: 0.0)
use_cim                   Use the CIM quantum sampler (requires QPU quota)
classifier_type           logistic / svm / random_forest (classifier mode)
```
