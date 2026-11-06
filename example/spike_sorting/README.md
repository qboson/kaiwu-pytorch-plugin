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
* `--zscore` - z-score every waveform before training,
* `--device cuda:0` - train on a GPU,
* `--epochs` - 100 matches the paper,
* `--realdata-dir` - add the TINS real recording as an extra dataset,
* `--max-spikes` - subsample spikes for quick experiments.

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

Single-seed reference run on the paper's four simulations plus the real
recording (100 epochs, batch size 128, latent size 10 for the VAE and the
QBM-VAE, AE code size 2, k-means with the ground-truth cluster count). Values
are ARI with AMI in parentheses.

| dataset | units | AE | VAE | QBM-VAE |
| --- | --- | --- | --- | --- |
| simulation 1 | 17 | 0.508 (0.754) | **0.663 (0.857)** | 0.281 (0.553) |
| simulation 4 | 5 | 0.587 (0.795) | **0.593 (0.806)** | 0.512 (0.751) |
| simulation 16 | 9 | 0.515 (0.765) | **0.628 (0.832)** | 0.583 (0.752) |
| simulation 35 | 13 | **0.707 (0.863)** | 0.636 (0.823) | 0.468 (0.699) |
| real recording | 83 | **0.094 (0.302)** | 0.058 (0.253) | 0.053 (0.226) |

Observations from this run:

* The AE/VAE baselines reproduce the paper's qualitative picture: clustering
  gets harder as the number of units grows, and the low-dimensional AE code is
  competitive but seed-sensitive.
* QBM-VAE is close to the baselines on simulations 4 and 16, but clearly behind
  on simulations 1 and 35, where the Boltzmann prior dominates the latent
  geometry.
* Longer QBM-VAE training degrades the features: on simulation 4 the ARI drops
  from 0.534 at 10 epochs to 0.512 at 100 epochs. Using `sigmoid(q)` instead of
  the raw encoder logits and stopping at 30 epochs gives the best QBM-VAE
  result (0.566), still below the VAE baseline.
* The real recording has 83 manually sorted units (~60 spikes per unit in a
  5000-spike subsample), so k-means ARI is low for every method; the paper also
  evaluates selected well-isolated units rather than all 83.

Improving the Boltzmann-prior branch (warm-up, early stopping, BM pretraining)
is follow-up work; the example keeps the paper's plain 100-epoch protocol so
the comparison stays comparable.

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
  reproduce a single run. The AE code layer is only two-dimensional and varies
  noticeably between seeds.
* Clustering uses the ground-truth cluster count, matching the paper's protocol;
  estimating the number of units is out of scope.
