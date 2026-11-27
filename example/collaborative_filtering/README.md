# Language: [中文](README_ZH.md) | [English](README.md)

# RBM Collaborative Filtering with Annealing Sampling

This example demonstrates a complete, modular collaborative filtering and top-K
recommendation pipeline built on `kaiwu.torch_plugin.RestrictedBoltzmannMachine`.
It combines sparsity-aware masked Contrastive Divergence with Ising annealing
negative-phase sampling (`rbm.sample(sampler)` + `rbm.objective(...)`), and
benchmarks the RBM recommender against classical **Popularity** and
**TruncatedSVD** baselines under a strict, leakage-free per-user holdout split.

## Algorithm Overview

A Restricted Boltzmann Machine (RBM) defines an energy function over item-level
visible units $v \in [0, 1]^V$ (normalized ratings) and binary latent preference
factors $h \in \{0, 1\}^H$:

$$E(v, h) = -a^\top v - b^\top h - v^\top W h$$

For sparse user-item interaction matrices with observation mask $M \in \{0, 1\}^{U \times V}$:

1. **Leakage-free prior imputation**: Unobserved entries ($M_{ui} = 0$) are filled
   during hidden-layer inference with damped per-item training means $\mu_i$
   computed strictly on the training split. Visible biases are initialized to
   $\log(\mu_i / (1 - \mu_i))$. This keeps positive hidden activations well-scaled
   across users with different rating counts and distinguishes low observed ratings
   ($\tilde{r}_{ui} = 0$) from unobserved items ($\mu_i \approx 0.5$).
2. **Masked Contrastive Divergence (CD-$k$)**: Positive and reconstructed negative
   energies are evaluated strictly on observed ratings ($M_{ui} = 1$), so unobserved
   items do not introduce spurious rating targets into $W$ or $a$.
3. **Ising Annealing Boltzmann Regularization**: At each epoch, the RBM is mapped
   to an Ising Hamiltonian via `RestrictedBoltzmannMachine.sample(sampler)` and
   combined with the masked CD objective via density-scaled
   `rbm.objective(s_pos, s_anneal)` (with parameter clipping via
   `rbm.clip_parameters(...)` to respect Ising hardware dynamic range).
4. **Top-K Recommendation**: Given a user's observed history, the model computes
   $h = p(h \mid v_{\text{in}})$ via `rbm.get_hidden` and reconstructs item
   ratings via `rbm.get_visible`, excluding all seen items before ranking.

## Directory Structure

```text
example/collaborative_filtering/
├── __init__.py       # Public package exports
├── dataset.py        # Deterministic synthetic generator, MovieLens-100K loader, per-user split
├── model.py          # RBMCollaborativeFiltering, RBMCFConfig, LocalIsingAnnealingSampler, build_sampler
├── baselines.py      # PopularityRecommender and bias-centered TruncatedSVDRecommender
├── metrics.py        # RMSE, MAE, Precision@K, Recall@K, NDCG@K (strict seen-item exclusion)
├── benchmark.py      # Benchmark orchestration, CLI parser, and table formatting
├── train_cf.py       # Runnable CLI entry point
├── README.md         # English documentation
└── README_ZH.md      # Chinese documentation
```

## Dependencies

This example uses **only core `kaiwu-torch-plugin` dependencies** (`torch`,
`numpy`, `kaiwu`) and requires **no external dataset downloads** in default mode.

```bash
pip install -r requirements/requirements.txt
pip install .
```

## Quick Start

Run the default offline synthetic benchmark (finishes in ~4 seconds on CPU):

```bash
python example/collaborative_filtering/train_cf.py
```

Quick smoke run (`--fast`) with JSON export:

```bash
python example/collaborative_filtering/train_cf.py --fast --output-json /tmp/cf_results.json
```

Compare classical Gibbs CD-1 ablation (`--sampler cd`):

```bash
python example/collaborative_filtering/train_cf.py --sampler cd
```

Run on a local MovieLens-100K `u.data` file (optional; download `ml-100k` locally):

```bash
python example/collaborative_filtering/train_cf.py \
  --dataset movielens \
  --data-path /path/to/ml-100k/u.data \
  --max-items 100 \
  --num-hidden 24 \
  --epochs 45
```

## Switching to Kaiwu SDK SA or Photonic Quantum CIM

`--sampler` selects the negative-phase Ising backend without changing any model code:

| `--sampler` | Backend | Requirements |
|---|---|---|
| `local_sa` (default) | `LocalIsingAnnealingSampler` (NumPy Metropolis-Hastings Ising SA) | Offline, no license required |
| `sa` | `kaiwu.classical.SimulatedAnnealingOptimizer` | Activated Kaiwu SDK license |
| `cim` | `kaiwu.cim.CIMOptimizer` + `PrecisionReducer` (`_kaiwu_compat`) | Activated Kaiwu SDK license + CIM QPU quota |
| `cd` | Classical Gibbs Contrastive Divergence only | Offline, no license required |

Example commands for licensed Kaiwu SDK / CIM environments (after initializing
your Kaiwu SDK license via `kw.license.init(user_id, sdk_code)`):

```bash
# Kaiwu SDK Simulated Annealing
python example/collaborative_filtering/train_cf.py --sampler sa

# Coherent Ising Machine (CIM) photonic quantum computer
python example/collaborative_filtering/train_cf.py \
  --sampler cim \
  --cim-task-name kpp_cf_rbm \
  --cim-task-mode sampling
```

For obtaining real CIM access and SDK authorization, see the repository root
[README.md](../../README.md).

## Quantitative Benchmark Results

Measured on CPU with the default synthetic latent-factor dataset (`--seed 42`,
400 users, 80 items, 6 latent preference clusters, 5,483 train / 788 val /
1,568 test interactions, 24.50% density; strict per-user holdout; top-10 ranking
over unseen items with relevance threshold $\ge 4.0$):

| Model | RMSE ↓ | MAE ↓ | P@10 ↑ | R@10 ↑ | NDCG@10 ↑ | CPU Time |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 1.1888 | 0.9786 | 0.0349 | 0.2044 | 0.1095 | 0.00 s |
| TruncatedSVD (`r=6`) | 1.0474 | 0.8427 | 0.0564 | 0.3292 | 0.1844 | 0.23 s |
| RBM-CF (`cd`, `H=24`) | 0.8413 | 0.6755 | 0.0720 | 0.4318 | 0.2170 | 1.39 s |
| RBM-CF (`local_sa`, `H=24`) | **0.8404** | **0.6749** | 0.0689 | 0.4087 | 0.2061 | 3.63 s |

Key observations:
- `RBM-CF (local_sa)` reduces test RMSE by **29.3%** compared to `Popularity`
  and **19.8%** compared to `TruncatedSVD (r=6)`, while improving `NDCG@10` by
  **+88.2%** over `Popularity` and **+11.8%** over `TruncatedSVD (r=6)`.
- During training, `RBM-CF (local_sa)` training RMSE decreases smoothly from
  `1.2259` (epoch 1) to `0.9070` (epoch 45) while exercising the full KPP
  `RestrictedBoltzmannMachine.sample(sampler)` Ising pipeline ready for `sa`/`cim`.

## Running Unit Tests

```bash
pytest tests/test_example_collaborative_filtering.py -v
```
