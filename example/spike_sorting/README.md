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

The simulations come from the public Kaggle datasets published by the paper's
first author:

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

## Environment

```bash
conda create -n quantum_env python=3.10
conda activate quantum_env
pip install -r requirements/requirements.txt
pip install .
pip install scikit-learn scipy
```

The Kaiwu SDK needs an authorization file before it can run a sampler. On the
first ``SimulatedAnnealingOptimizer`` call the SDK asks for the *user ID* and
*SDK code* obtained from <https://platform.qboson.com/> and caches them in
`license.lic`. Tests that mock the sampler keep running without a license, but
the experiments below require one.

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

* `--latent-dim` – latent size of the VAE/QBM-VAE (default 10; the RBM prior
  splits it into two halves, so keep it even),
* `--zscore` – z-score every waveform before training,
* `--device cuda:0` – train on a GPU,
* `--epochs` – 100 matches the paper.

The run writes `results/metrics.csv` and `results/metrics.json` with ARI, AMI,
V-measure (external) and Calinski-Harabasz, Davies-Bouldin, Silhouette
(internal) scores for every (simulation, model) pair.

## Protocol

1. Load the recording and cut 79-sample waveforms at `spike_first_sample`.
2. Align the positive peak to sample 39 (optionally z-score).
3. Train the feature extractor on the waveforms only (labels are never used).
4. Take the code layer (`ae`), the posterior mean (`vae`) or the encoder logits
   (`qbmvae`) as features.
5. Run k-means with `n_clusters = number of ground-truth units` (the multi-unit
   cluster counts as one cluster, as in the paper).
6. Report the six clustering metrics.

## Files

| file | purpose |
| --- | --- |
| `spike_data.py` | MATLAB loading, waveform extraction, peak alignment |
| `models.py` | AE / Gaussian VAE baselines and the QBM-VAE wrapper |
| `metrics.py` | k-means clustering and the six reported metrics |
| `run_experiment.py` | CLI that trains, evaluates and writes CSV/JSON |

## Notes and limitations

* The real-data (`realdata`) archive uses the TINS `.ssd` container; a loader is
  planned as a follow-up. The simulation table is the paper's main result.
* QBM-VAE training uses simulated annealing by default so it runs on a laptop;
  pass a CIM sampler to compare against real quantum hardware.
* The paper reports per-variant averages over several runs; use `--seed` to
  reproduce a single run.
