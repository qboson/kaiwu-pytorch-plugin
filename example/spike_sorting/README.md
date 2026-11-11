# Spike Sorting: Autoencoder Feature Extraction with QBM-VAE

This example reproduces the spike-sorting feature-extraction study of

> Ardelean, E.-R., Coporîie, A., Ichim, A.-M., Dînşoreanu, M., & Mureşan, R. C.
> (2023). *A study of autoencoders as a feature extraction technique for spike
> sorting.* PLOS ONE, 18(3), e0282810.
> https://doi.org/10.1371/journal.pone.0282810

and replaces the paper's autoencoder/VAE baselines with the plugin's
**QBM-VAE** (`kaiwu.torch_plugin.QVAE`, i.e. a VAE whose latent prior is a
restricted Boltzmann machine sampled with the Kaiwu SDK).

## Why this example

Spike sorting groups action potentials by the neuron that generated them. The
paper shows that the *feature extraction* step dominates clustering quality and
compares ten autoencoder variants against PCA/ICA/Isomap on 79-sample spike
waveforms. This example keeps the same protocol but adds a quantum-Boltzmann
latent, which is the plugin's core scientific contribution.

## Data

The datasets are the public Kaggle ones published by the paper's first author:

* simulations: <https://www.kaggle.com/datasets/ardeleanrichard/simulationsdataset>
  (`simulation_<n>.mat`, `ground_truth.mat`)
* real recording: <https://www.kaggle.com/datasets/ardeleanrichard/realdata>

The paper evaluates simulations **1, 4, 16 and 35** (16, 4, 8 and 12 units plus
the multi-unit cluster) and uses simulation 14 to illustrate alignment. Only
those files are needed for the main table.

Each `simulation_<n>.mat` stores a 24 kHz, 10-minute recording under `data`;
`ground_truth.mat` stores `spike_first_sample`, `spike_classes` and
`su_waveforms` cell arrays. The loader extracts 79 samples per spike, aligns the
positive peak to index 39 (the paper's `align_to_peak=2` mode) and optionally
z-scores every waveform.

The real recording is the TINS "Spike Sorted Dataset" (`.ssd`): a text
descriptor holding unit names, electrodes and spike counts, plus binary
companions. `load_realdata` reads the 58-sample float32 waveforms from
`*.ssduw` and rebuilds the reference unit labels from the descriptor. Those
labels come from the manual sorting shipped with the dataset, so they are a
reference partition rather than a synthetic ground truth.

## Environment

```bash
conda create -n quantum_env python=3.10
conda activate quantum_env
pip install -r requirements/requirements.txt
pip install .
pip install scikit-learn scipy
```

The Kaiwu SDK needs an authorization file before it can run a sampler. On the
first `SimulatedAnnealingOptimizer` call the SDK asks for the *user ID* and
*SDK code* obtained from <https://platform.qboson.com/> and caches them in a
`license.lic` file. Tests that mock the sampler keep running without a license,
but the QBM-VAE experiments below require one; the AE/VAE baselines are plain
PyTorch and do not.

## Usage

```bash
python example/spike_sorting/run_experiment.py \
    --data-dir /path/to/simulations \
    --simulations 1 4 16 35 \
    --models ae vae qbmvae \
    --epochs 100 \
    --out example/spike_sorting/results
```

Useful flags:

* `--latent-dim` - latent size of the VAE/QBM-VAE (default 10; the RBM prior
  splits it into two halves, so keep it even),
* `--representation {q,sigmoid}` - QBM-VAE feature space; `sigmoid` (default)
  clusters better at the tuned 30-epoch budget,
* `--epochs` - 100 matches the paper; the tuned QBM-VAE is best at 30,
* `--zscore` - z-score every waveform before training,
* `--device cuda:0` - train on a GPU,
* `--realdata-dir` - add the TINS real recording as an extra dataset,
* `--max-spikes` - subsample the real recording for quick experiments.

The run writes `results/metrics.csv` and `results/metrics.json` with ARI, AMI,
V-measure (external) and Calinski-Harabasz, Davies-Bouldin, Silhouette
(internal) scores for every (dataset, model) pair.

## Protocol

1. Load the recording and cut 79-sample waveforms at `spike_first_sample`.
2. Align the positive peak to sample 39 (optionally z-score).
3. Train the feature extractor on the waveforms only (labels are never used).
4. Take the code layer (`ae`), the posterior mean (`vae`) or the encoder logits
   (`qbmvae`) as features.
5. Run k-means with `n_clusters = number of ground-truth units` (the multi-unit
   cluster counts as one cluster, as in the paper).
6. Report the six clustering metrics.

## Results

All numbers are clustering ARI on the paper's protocol: 79-sample waveforms,
peak aligned to sample 39, k-means with the ground-truth cluster count. The
baselines use the paper's configuration (AE code size 2, Gaussian VAE with a
10-dimensional latent); the QBM-VAE uses the tuned configuration derived below
(latent 10, 30 epochs, `sigmoid(q)` features).

### Equal training budget (30 epochs)

Means over five seeds for the baselines and three seeds for the QBM-VAE. The
QBM-VAE needs a second optimizer pass and one Ising sample per batch, so an
epoch-budget comparison is the fair one.

| dataset | units | AE (ARI) | VAE (ARI) | QBM-VAE (ARI) |
| --- | --- | --- | --- | --- |
| simulation 1 | 17 | **0.631 ± 0.078** | 0.532 ± 0.023 | 0.413 ± 0.170 |
| simulation 4 | 5 | 0.476 ± 0.114 | **0.560 ± 0.005** | 0.472 ± 0.084 |
| simulation 16 | 9 | 0.563 ± 0.122 | **0.618 ± 0.044** | 0.511 ± 0.073 |
| simulation 35 | 13 | **0.634 ± 0.055** | 0.606 ± 0.022 | 0.576 ± 0.041 |

### Longer baselines (100 epochs)

The paper trains its autoencoders for 100 epochs; the baselines improve further
and the gap to QBM-VAE widens.

| dataset | units | AE (ARI) | VAE (ARI) |
| --- | --- | --- | --- |
| simulation 1 | 17 | 0.664 ± 0.098 | 0.642 ± 0.050 |
| simulation 4 | 5 | 0.702 ± 0.140 | 0.566 ± 0.026 |
| simulation 16 | 9 | 0.643 ± 0.067 | 0.651 ± 0.040 |
| simulation 35 | 13 | 0.654 ± 0.045 | 0.561 ± 0.043 |

### Real recording (5000-spike subsample)

83 manually sorted units, so every method has a low k-means ARI; the paper also
evaluates selected well-isolated units rather than all 83.

| dataset | AE (ARI) | VAE (ARI) | QBM-VAE (ARI) |
| --- | --- | --- | --- |
| real recording | 0.076 ± 0.013 | 0.068 ± 0.010 | 0.053 |

### Tuning the QBM-VAE

Training length and feature representation dominate the result. On simulation 4
(latent 10):

| epochs | q (logits) | sigmoid(q) |
| --- | --- | --- |
| 10 | 0.505 | 0.422 |
| 30 | 0.531 | **0.566** |
| 60 | 0.503 | **0.566** |
| 100 | 0.508 | 0.464 |

Latent size (simulation 4, 30 epochs, `sigmoid(q)`):

| latent | 6 | 8 | 10 | 16 | 20 |
| --- | --- | --- | --- | --- | --- |
| ARI | 0.404 | 0.387 | **0.566** | 0.545 | 0.562 |

Simulation 16 shows the same epoch trend (10 epochs 0.589, 30 epochs 0.589,
100 epochs 0.518).

Findings:

* Longer training hurts the QBM-VAE: the Boltzmann prior slowly reshapes the
  latent geometry until the k-means features degrade (simulation 4: 0.566 at
  30 epochs vs 0.464 at 100).
* `sigmoid(q)` is a better feature space than the raw encoder logits at the
  peak of the curve (+0.035 on simulation 4 at 30 epochs), but it needs enough
  training to be useful (10 epochs: 0.422 vs 0.505).
* A 10-dimensional latent is the sweet spot; 6 and 8 collapse, 16 and 20 tie
  without a clear gain.
* Stronger KL regularisation (`kl_beta=1e-2`) collapses the features (0.378 on
  simulation 4) and a smaller BM learning rate (`1e-4`) does not rescue the
  60-epoch result (0.527).
* With the tuned configuration the QBM-VAE improves a lot over the default
  100-epoch/raw-logit run (simulation 1: 0.281 -> 0.529; simulation 35:
  0.468 -> 0.579) but, averaged over seeds, it still does not beat the classical
  baselines. We report that negative result rather than cherry-picking a seed.

Improving the Boltzmann-prior branch (warm-up, early stopping on the
reconstruction loss, BM pretraining) is follow-up work.

## Files

| file | purpose |
| --- | --- |
| `spike_data.py` | MATLAB loading, waveform extraction, peak alignment, `.ssd` loader |
| `models.py` | AE / Gaussian VAE baselines and the QBM-VAE wrapper |
| `metrics.py` | k-means clustering and the six reported metrics |
| `run_experiment.py` | CLI that trains, evaluates and writes CSV/JSON |

## Notes and limitations

* QBM-VAE training uses simulated annealing by default so it runs on a laptop;
  pass a CIM sampler to compare against real quantum hardware.
* The paper reports per-variant averages over several runs; use `--seed` to
  reproduce a single run. All tables above average over seeds; the AE code layer
  is only two-dimensional and varies noticeably (up to ±0.16 ARI), and the
  QBM-VAE is seed-sensitive too.
* The tuned QBM-VAE closes most of the gap to the baselines but does not beat
  them on these datasets. This is reported as a negative result: the example
  documents the tuning evidence instead of only the best-performing seed.
* Clustering uses the ground-truth cluster count, matching the paper's protocol;
  estimating the number of units is out of scope.
