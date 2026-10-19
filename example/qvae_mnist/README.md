# KPP QVAE MNIST Image Generation

This directory provides a Quantum Variational Autoencoder (Q-VAE)
implementation for image generation on the MNIST dataset using
`kaiwu-pytorch-plugin`. The workflow covers training a Q-VAE whose latent
prior is an RBM sampled by the Kaiwu SDK (classical SA or quantum CIM),
generating new digit images, evaluating generation quality with FID, and
using the learned latent representations for downstream classification.

## Dependencies

```bash
# Core dependencies (run from the repository root)
pip install -r requirements/requirements.txt   # kaiwu==1.3.1, torch==2.7.0, numpy==2.2.6
pip install .                                  # install the plugin

# Example-side dependencies
pip install -r requirements/requirements_example.txt
```

The notebooks additionally use `pandas`, `tqdm`, `pillow`, `imageio`, and
`gif` (the latter only for the training-evolution animation). Install them
if they are missing:

```bash
pip install pandas tqdm pillow imageio gif
```

## File Structure

```text
qvae_mnist/
├── model/
│   ├── config.py                  # Config: unified hyperparameters
│   ├── model.py                   # MnistQVAE (RBM-based latent prior)
│   ├── networks.py                # BasicEncoder / BasicDecoder
│   └── feature_extractor.py       # FeatureExtractor (latent 'q' / 'zeta')
├── trainer/
│   ├── trainer.py                 # Trainer: data loading, training loop, saving
│   └── model_tuner.py             # ModelTuner: optimizers and training steps
├── utils/
│   ├── loadMNIST.py               # MNIST / Fashion-MNIST / KMNIST loaders
│   ├── helpers.py                 # generation, t-SNE and FID evaluation helpers
│   ├── logging.py
│   └── exception.py
├── downstream/
│   ├── classifier.py              # MLPClassifier on Q-VAE features
│   └── pipeline.py                # get_full_pipeline: sklearn pipeline
├── train_qvae.ipynb               # Train + generate + FID evaluation
├── train_qvae_classifier.ipynb    # Train + feature extraction + MLP classification
├── run_pipeline.py                # Command-line entry: QVAE + MLP pipeline
└── run_pipeline.sh                # Example CLI invocation
```

## Datasets

`name` supports `mnist` (default), `fashion-mnist`, and `kmnist`; the data
is downloaded automatically to `data_path` on first use.

## Training

Run the notebooks from inside `example/qvae_mnist/` (they import the local
`model`, `trainer`, `utils`, and `downstream` packages).

### Option A: Notebook

1. **Train the Q-VAE** — open `train_qvae.ipynb` and run the cells. This
   trains `MnistQVAE` for the configured number of epochs, saves
   reconstructions and the training curve, optionally produces a final
   t-SNE plot (`run_tsne=True`) and a training-evolution animation
   (`generate_animation=True`), then generates a grid of new images and
   evaluates the FID score.
2. **Train Q-VAE + downstream classifier** — open
   `train_qvae_classifier.ipynb`. It trains the Q-VAE, extracts latent
   features (`feature_type='q'`), trains an MLP classifier on them, and
   reports test accuracy; the last cell runs the same workflow through the
   end-to-end `get_full_pipeline()`.

### Option B: Command Line

```bash
python run_pipeline.py \
  --model-type QVAE \
  --name mnist \
  --data-path ./data \
  --batch-size 256 \
  --epochs 10 \
  --lr 8e-4 \
  --bm-lr 8e-4 \
  --sampler-type sa \
  --loss-type bernoulli \
  --feature-type q \
  --mlp-hidden-dims 256 128 \
  --mlp-output-dim 10 \
  --mlp-lr 8e-5 \
  --mlp-epochs 100 \
  --compute_energy
```

`bash run_pipeline.sh` is equivalent to the command above.

## Outputs

Results are written under `--output-dir` (or
`./output/<TYPE>_<timestamp>` by default):

```text
qvae_training_curve.png          # Train/test loss curves
reconstruction_epoch_*.png       # Original vs reconstructed images (every ~10% of epochs)
QVAE_t-SNE_epochs_*.png          # t-SNE of the latent space (when --run-tsne)
qvae_training_evolution.gif      # t-SNE animation over epochs (notebook)
qvae_energy_by_class.png         # BM energy distribution per class (when --compute_energy)
final_QVAE                       # Final model checkpoint
generated_x.png                  # Grid of generated images (notebook)
fid_results.txt                  # FID evaluation result (notebook)
results.json                     # Config + losses, or test accuracy (CLI)
```

## Evaluation

- **Generation quality (FID)**: `train_qvae.ipynb` computes the FID score
  between generated and real test images (flattened 28x28 inputs) using
  `torchmetrics` and writes the result to `fid_results.txt`.
- **Classification accuracy**: `train_qvae_classifier.ipynb` and
  `run_pipeline.py` report the test accuracy of the MLP classifier trained
  on Q-VAE latent features.

## Key Parameters

```text
--name                 Dataset: mnist / fashion-mnist / kmnist (default: mnist)
--data-path            Where to store/download data (default: ./data)
--num-latent-units     Latent dimensionality, split half/half into RBM
                       visible/hidden units (default: 256)
--dist-beta            Exponential smoothing scale for latent samples (default: 10.0)
--kl-beta              KL weight (default: 1e-6)
--sampler-type         sa or cim (default: sa)
--loss-type            bernoulli or mse (default: bernoulli)
--weight-decay         Weight decay (default: 0.01)
--batch-size           Batch size (default: 256)
--epochs               Number of epochs (default: 50; notebooks use 10)
--lr                   VAE learning rate (default: 8e-4)
--bm-lr                RBM learning rate (default: 8e-4)
--use-cuda             Enable CUDA when available
--feature-type         Latent features used downstream: q or zeta (default: q)
--run-tsne             Generate a final t-SNE plot after training
--compute_energy       Compute and plot per-class BM energies
--mlp-hidden-dims      MLP classifier hidden dims (default: 256 128)
--mlp-output-dim       MLP output dim (default: 10)
--mlp-lr / --mlp-batch-size / --mlp-epochs / --mlp-weight-decay
                       MLP classifier training hyperparameters
```
