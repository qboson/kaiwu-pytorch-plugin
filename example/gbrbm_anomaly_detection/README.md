# Language Versions: [中文](README_ZH.md) | [English](README.md)

# `example/gbrbm_anomaly_detection`

This directory provides an end-to-end **unsupervised anomaly detection** example for
continuous features built on the Gaussian-Bernoulli Restricted Boltzmann Machine (GBRBM):
the model is trained on normal samples only, samples are scored with the free energy and a
deterministic reconstruction error, and the result is compared against a classical
`IsolationForest` on exactly the same held-out split.

## Overview

- **Zero core changes**: nothing under `src/kaiwu/torch_plugin/` is modified. The case only
  uses the public API of `GaussianBernoulliRestrictedBoltzmannMachine`
  (`marginal_energy`, `infer_from_gaussian`, `infer_from_bernoulli`, `sample`, `objective`).
- **Leakage-free by construction**: `train_normal` / `val` / `test` are split on raw samples
  first, and the `StandardScaler` is fitted **only** on the normal training rows. No
  validation or test value influences the scaling statistics.
- **One score direction**: both scores satisfy "higher means more anomalous", so they can be
  passed straight to `roc_auc_score` and `average_precision_score`.
- **Local sampling by default**: the default sampler is `SimulatedAnnealingOptimizer`, which
  never submits a cloud task. A `--sampler cim` entry point is kept ready for licensed
  hardware and stays compatible with Kaiwu SDK 1.3.1-1.4.3.
- **Non-blocking plots**: the plotting module uses `Figure` + `FigureCanvasAgg` only, so it
  never opens a GUI window or calls `plt.show()`; the main entry point degrades gracefully
  when `matplotlib` is not installed.

## Layout

```text
example/gbrbm_anomaly_detection/
├── ad_config.py               # frozen dataclass configuration and result containers
├── ad_data.py                 # data loading, leakage-free split and standardization
├── ad_sampler.py              # local SA default, CIM compatibility, license env vars
├── ad_scorer.py               # GBRBMAnomalyScorer (free energy / reconstruction)
├── ad_trainer.py              # unsupervised training loop, device and seed management
├── ad_eval.py                 # threshold selection, test metrics, baseline, JSON export
├── ad_plot.py                 # pure OO Agg canvas, non-blocking PNG export
├── run_anomaly_detection.py   # CLI entry point and orchestration
├── requirements.txt           # extra dependencies of this case
└── README_ZH.md / README.md   # bilingual documentation
```

## Dataset and leakage-free split

The data comes from `sklearn.datasets.load_breast_cancer()` (Breast Cancer Wisconsin
Diagnostic): 569 samples with 30 continuous cell-nucleus features, bundled with
`scikit-learn`, so no download is required.

Label convention: **benign (`target == 1`, 357 rows) is the normal class `0`; malignant
(`target == 0`, 212 rows) is the anomaly class `1`**.

| Split | Rows | Normal | Anomaly | Purpose |
| --- | ---: | ---: | ---: | --- |
| `train_normal` | 214 | 214 | 0 | Unsupervised training of GBRBM and `IsolationForest` |
| `val` | 177 | 71 | 106 | Threshold selection |
| `test` | 178 | 72 | 106 | Final reporting only |

Split procedure (`ad_data.py`):

1. Take all normal indices and cut a normal-only training set plus held-out normal rows with
   `train_normal_ratio=0.6`.
2. Merge the held-out normal rows with all anomaly rows and split them into `val` / `test`
   with `stratify` and `val_ratio_of_eval=0.5`, so both splits keep a similar anomaly rate.
3. Only after the split, create `StandardScaler` and call `fit(features[train_indices])` —
   nothing else.
4. Transform the three subsets with that single scaler and cast them to `float32`.

All indices are kept in `AnomalyDatasetSplit.train_indices / val_indices / test_indices`,
which lets the unit tests assert that the three sets are pairwise disjoint and cover every
row.

## Anomaly scoring

`GBRBMAnomalyScorer` (`ad_scorer.py`) calls public APIs only and aligns every input to
`(model.device, model.dtype)`. With continuous input $v \in \mathbb{R}^{D}$, hidden state
$h \in \{0,1\}^{H}$, mean $\mu$, variance $\sigma^2 = \exp(\text{log\_var})$, weights $W$ and
hidden bias $b$:

### 1. Free energy (default primary score)

```text
F(v) = 0.5 * Σ_i (v_i - μ_i)² / σ_i²  -  Σ_j softplus( Σ_i v_i W_ij / σ_i² + b_j )
```

`marginal_energy` sums out the hidden units analytically under `torch.no_grad()`, giving the
exact free energy. Since $p(v) \propto \exp(-F(v))$, a low-probability anomaly naturally
obtains a higher $F(v)$ — **no sign flip is needed** — and the quadratic term drives
$F(v) \to +\infty$ as $\lVert v \rVert \to \infty$.

### 2. Deterministic reconstruction error (secondary score)

```text
p(h=1|v) = sigmoid( v W / σ² + b )        # infer_from_gaussian(binarize=False)
v̂        = p(h=1|v) Wᵀ + μ                # infer_from_bernoulli(no_random=True)
s_RE(v)  = (1/D) * Σ_i (v_i - v̂_i)²
```

The encoder keeps the conditional activation probability and the decoder uses the
conditional mean, so this score carries **no sampling noise** at inference time and is fully
repeatable; a larger reconstruction error means the sample lies farther from the normal
manifold.

Both scores share one convention: **higher means more anomalous**.

## Training

`ad_trainer.train_gbrbm()` receives the normal training rows only:

1. `resolve_device()` resolves `auto` / `cpu` / `cuda` and raises when `cuda` is requested
   without a GPU instead of silently falling back.
2. `set_global_seed()` fixes Python `random`, `np.random` and `torch.manual_seed`.
3. The GBRBM is built with `dtype=torch.float32` and the resolved device, and
   `model.log_var.requires_grad_(False)` runs when `freeze_log_var=True` (the default).
4. Every mini-batch performs a contrastive-divergence style update:
   - positive phase `s_pos = infer_from_gaussian(batch, binarize=False)`;
   - negative phase `s_neg = model.sample(sampler)`, where the sampler minimizes the
     Bernoulli-side Ising model;
   - objective `loss = objective(s_pos, s_neg) + weight_decay * (‖W‖² + ‖b‖²)`;
   - backpropagation, gradient-norm clipping and an SGD step.

**Why `log_var` is frozen by default**: `GBRBM.sample()` always uses
`infer_from_bernoulli(..., no_random=True)`, so the negative Gaussian state is the conditional
mean without the $\mathcal{N}(0, \sigma^2)$ noise term. For $\mu, W, b$ this is an unbiased
Rao-Blackwellized estimate, but for `log_var` the true expectation
$\mathbb{E}_{v|h}[(v_i-\mu_i)^2] = ((Wh)_i)^2 + \sigma_i^2$ contains a missing $+\sigma_i^2$
term, which biases $\partial\mathcal{L}/\partial\text{log\_var}$ and can make the variance
drift. Following Hinton (2010), the case keeps $\sigma_i^2 = 1$ inside the standardized space
(zero mean, unit variance) and exposes `ModelConfig.freeze_log_var` as an explicit switch.

## Samplers: local SA (default) and the CIM entry point

`ad_sampler.build_sampler()`:

| `--sampler` | Behaviour |
| --- | --- |
| `sa` (default) | `kaiwu.classical.SimulatedAnnealingOptimizer`, runs locally, no cloud task |
| `cim` | `kaiwu.cim.CIMOptimizer`, wrapped by `PrecisionReducer` by default, requires a licensed Kaiwu account |

The CIM entry point normalizes the Kaiwu 1.3.1 and 1.4.0-1.4.3 differences through
`normalize_task_mode()` and `PrecisionReducer` from `kaiwu.torch_plugin._kaiwu_compat`:

| `--cim-task-mode` | Kaiwu 1.3.1 | Kaiwu 1.4.x |
| --- | --- | --- |
| `sampling` / `sample` | `"sample"` | `TaskMode.SAMPLING` |
| `optimization` / `quota` | `"quota"` | `TaskMode.OPTIMIZATION` |

### License configuration (never hardcode secrets)

`ad_sampler.init_kaiwu_license_from_env()` reads a credential pair in priority order and calls
`kw.license.init(user_id, sdk_code)` only when both values are non-empty:

1. `USER_ID` + `SDK_CODE`
2. `LICENSE_USER_ID` + `LICENSE_SDK_CODE`

```bash
# Linux / macOS
export USER_ID="<your user id>"
export SDK_CODE="<your sdk code>"
```

```powershell
# Windows PowerShell
$env:USER_ID = "<your user id>"
$env:SDK_CODE = "<your sdk code>"
```

Without credentials the script neither forges nor bypasses any authorization check; it keeps
the ambient SDK license state, and the licensed annealing call is still gated by the Kaiwu SDK
itself. Never commit credentials to code, example configs or the repository.

## Quick start

```bash
pip install -e .                                       # install the plugin from the repo root
pip install -r example/gbrbm_anomaly_detection/requirements.txt

# Seconds-long smoke run: 2 epochs, 8 hidden units, lighter SA schedule
python example/gbrbm_anomaly_detection/run_anomaly_detection.py --quick --seed 42 --device cpu

# Full reference configuration: 30 epochs, 16 hidden units
python example/gbrbm_anomaly_detection/run_anomaly_detection.py --seed 42 --device cpu \
    --output-dir example/gbrbm_anomaly_detection/outputs/full

# Skip figures (also skipped automatically when matplotlib is unavailable)
python example/gbrbm_anomaly_detection/run_anomaly_detection.py --quick --no-plot
```

## CLI options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--epochs` | int | `30` | Training epochs; forced to `2` by `--quick` |
| `--seed` | int | `42` | Global seed for the split, PyTorch, SA and `IsolationForest` |
| `--output-dir` | path | `example/gbrbm_anomaly_detection/outputs` | Target directory of `metrics.json` and the two PNG figures |
| `--device` | str | `auto` | `auto` / `cpu` / `cuda` (fails fast when CUDA is unavailable) |
| `--quick` | flag | off | Fast mode: `epochs=2`, `num_hidden=8`, `sa_size_limit=16`, `sa_iterations_per_t=3`, `sa_alpha=0.9` |
| `--sampler` | str | `sa` | `sa` (local simulated annealing) / `cim` (coherent Ising machine entry point) |
| `--score-method` | str | `free_energy` | Score used for the histogram: `free_energy` / `reconstruction`; both are always reported in JSON |
| `--num-hidden` | int | `16` | Number of Bernoulli hidden units |
| `--batch-size` | int | `32` | Mini-batch size |
| `--lr` | float | `5e-3` | SGD learning rate |
| `--weight-decay` | float | `1e-2` | L2 regularization factor |
| `--no-plot` | flag | off | Write `metrics.json` only, skip figures |
| `--cim-task-name` | str | `gbrbm_anomaly_detection` | CIM task name (only with `--sampler cim`) |
| `--cim-task-mode` | str | `sampling` | `sampling` / `sample` / `optimization` / `quota` |
| `--cim-no-precision-reducer` | flag | off | Disable the CIM `PrecisionReducer` wrapper |

## Outputs

```text
<output-dir>/
├── metrics.json              # config, split statistics, training history, three metric groups
├── score_distribution.png    # test score histogram with the validation threshold
└── roc_curve_comparison.png  # ROC comparison of both GBRBM scores and IsolationForest
```

The `metrics` object in `metrics.json` contains `gbrbm_free_energy`,
`gbrbm_reconstruction` and `isolation_forest`, each with
`roc_auc / pr_auc / f1 / precision / recall / val_f1 / threshold`.

## Reference results

Full configuration (`--seed 42 --device cpu`, 30 epochs, 16 hidden units) on 178 test rows
(106 anomalies), with the threshold selected on validation and applied unchanged to test:

| Model / score | ROC-AUC | PR-AUC | F1 | Precision | Recall | Validation threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GBRBM (free energy) | 0.9434 | 0.9622 | 0.8789 | 0.8376 | 0.9245 | 10.9980 |
| GBRBM (reconstruction) | 0.9434 | 0.9620 | 0.8829 | 0.8448 | 0.9245 | 1.4692 |
| `IsolationForest` | 0.9450 | 0.9593 | 0.8826 | 0.8785 | 0.8868 | -0.0043 |

Training trace: mean objective `15.0235 → 14.5062`, normal-training free energy
`3.9229 → 3.5606`.

The `--quick` smoke mode (2 epochs, 8 hidden units) also runs the full pipeline and reaches
ROC-AUC `0.9441` / PR-AUC `0.9626` / F1 `0.8778` with the free-energy score.

Note that SA and CIM are stochastic samplers: the Ising solution quality depends on the
temperature schedule, `size_limit` and SDK version, so the numbers above vary slightly across
runs, Kaiwu versions and hardware. `IsolationForest` is fully reproducible with
`random_state=42`.

## Third-party dependencies and licenses

The case adds no new heavy dependency; everything is already covered by the repository
requirements and the `NOTICE` file and stays within the CI whitelist
`pip-licenses --allow-only "MIT;LGPL;BSD;MPL;Apache;Unlicense;PSF"`.

| Dependency / data | Purpose | License |
| --- | --- | --- |
| `torch` | GBRBM tensor math, training and inference | BSD-3-Clause |
| `numpy` | Numeric arrays and metric pre-processing | BSD-3-Clause |
| `kaiwu` | `SimulatedAnnealingOptimizer` / `CIMOptimizer` / `PrecisionReducer` | Kaiwu SDK proprietary (existing plugin dependency) |
| `scikit-learn` | Dataset, split, `StandardScaler`, `IsolationForest`, ROC/PR/F1 | BSD-3-Clause |
| `matplotlib` (optional) | `FigureCanvasAgg` PNG export | PSF-based (Matplotlib License) |
| Breast Cancer Wisconsin (Diagnostic) | 569 rows x 30 continuous features benchmark | CC BY 4.0 (bundled with `scikit-learn`) |
