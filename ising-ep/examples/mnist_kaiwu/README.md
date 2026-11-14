# MNIST/100 Ising EP experiment

[English](README.md) · [中文](README.zh-CN.md)

Trains the 784-64-40 Ising equilibrium-propagation network on MNIST/100 using
Kaiwu's simulated annealer for both relaxation phases. For the algorithm itself
see the [project README](../../README.md); this file covers running the
experiment and what the archived run produced.

## Data

MNIST/100 is the reduced split used in the Ising EP literature: the **first 100
training and first 10 test images of each digit, in original IDX file order**.
Selecting by position rather than at random makes the split reproducible from the
source archives alone, with no seed to record. Pixels are scaled to `[0, 1]` and
flattened in C order; there is no augmentation.

Place the four archives in `data/MNIST/raw/`:

```
train-images-idx3-ubyte.gz   train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz    t10k-labels-idx1-ubyte.gz
```

Or pass `--download-data` to fetch them on first use. Downloading is opt-in so a
training run never reaches the network unexpectedly; without the flag a missing
archive raises `FileNotFoundError` rather than silently inventing a subset.

## Commands

These assume the package is installed (`pip install -e ".[dev]"` from the
project root) so that `ising_ep` is importable — see
[Setup](../../README.md#setup). Run them from the project root.

Smoke run — verifies the whole path end to end in a few minutes:

```powershell
python -m examples.mnist_kaiwu.experiment `
  --output runs/smoke --seeds 0 --n-hidden 8 --epochs 1 `
  --candidate-count 2 --iterations-per-t 4
```

The run that produced `results/primary/`:

```powershell
python -m examples.mnist_kaiwu.experiment `
  --output runs/primary --seeds 0,1,2 `
  --n-hidden 64 --epochs 5 --batch-size 10 --lr 0.01 --beta 1.0
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `--output` | required | Directory receiving artifacts |
| `--data-root` | `data` | Directory containing `MNIST/raw` |
| `--download-data` | off | Fetch missing archives instead of failing |
| `--seeds` | `0,1,2` | Comma-separated experiment seeds |
| `--n-hidden` | `64` | Width of a single hidden layer |
| `--hidden-sizes` | unset | Comma-separated hidden widths, e.g. `128,64`; overrides `--n-hidden` |
| `--epochs` | `5` | Passes over the training split |
| `--batch-size` | `10` | Samples sharing one parameter state per update |
| `--lr` | `0.01` | SGD learning rate, used for every layer unless overridden |
| `--lr-weights` | unset | Weight learning rates, one per connection (`len(hidden) + 1` values) |
| `--lr-biases` | unset | Bias learning rates, one per spin layer (`len(hidden) + 1` values) |
| `--init-gains` | unset | Initial-range scale per connection; default `1` |
| `--weight-decay` | `0.0` | L2 decay on weight matrices only |
| `--beta` | `1.0` | Nudging coefficient, must be positive |
| `--initial-temperature` | `1.25` | Anneal start temperature |
| `--alpha` | `0.95` | Geometric cooling factor |
| `--cutoff-temperature` | `0.001` | Anneal stop temperature |
| `--iterations-per-t` | `10` | Sweeps per temperature |
| `--candidate-count` | `8` | Candidates returned per solve |

Cost scales as `2 × n_train × epochs + n_test` solver calls, so the default run
issues 10,100 calls per seed. Lowering `--candidate-count` and
`--iterations-per-t` is the fastest way to shorten a smoke run.

## Artifacts

```
<output>/config.json              Full configuration, including the anneal schedule
<output>/summary.json             Cross-seed mean/std/min/max
<output>/seeds/<seed>/
    initial-model.npz             weights.N, biases.N before training
    checkpoint.npz                Same after the final epoch
    epochs/<e>/model.npz          Per-epoch parameter snapshot
    epochs/<e>/dynamics.json      hidden_flip, output_flip, zero_update, ...
    test/metrics.json             accuracy, tie_rate, confusion, per_class
    test/predictions.json         One record per test image
    status.json                   Seed summary, learning evidence, call counts
```

JSON is written with sorted keys and no insignificant whitespace, so repeated
runs of one configuration produce byte-identical files that diff cleanly.

### Reading `dynamics.json`

| Field | Meaning |
|---|---|
| `hidden_flip` | Fraction of hidden spins that differed between phases |
| `output_flip` | Same for output spins |
| `zero_update` | Fraction of samples where both phases agreed entirely — these contribute exactly zero gradient |
| `hidden_zero_update` | Fraction where the hidden state alone was unchanged |

A `zero_update` near 1.0 means the nudge is too weak or the anneal budget too
small: the run will complete without error but learn nothing. `status.json`'s
`learning_evidence.passed` checks for exactly that failure mode by requiring both
couplings to have moved and at least one epoch to have flipped hidden spins on
≥5% of samples.

## Archived results

From [`results/primary/`](../../results/primary). 784-64-40, 5 epochs, batch 10,
`lr = 0.01`, `β = 1.0`, seeds 0/1/2, default anneal schedule.

### Per seed

| Seed | Accuracy | Tie rate | ‖ΔW‖ | ‖ΔV‖ | Hidden flip rate by epoch |
|---|---|---|---|---|---|
| 0 | 0.15 | 0.62 | 2.493 | 2.657 | 0.685, 0.412, 0.318, 0.240, 0.232 |
| 1 | 0.54 | 0.06 | 3.498 | 3.349 | 0.930, 0.829, 0.664, 0.612, 0.524 |
| 2 | 0.46 | 0.18 | 3.000 | 3.049 | 0.885, 0.728, 0.635, 0.551, 0.522 |

### Cross-seed

| Metric | mean | std | min | max |
|---|---|---|---|---|
| Accuracy | 0.3833 | 0.1682 | 0.15 | 0.54 |
| Tie rate | 0.2867 | 0.2407 | 0.06 | 0.62 |

### Per class

| Class | mean | std | min | max |
|---|---|---|---|---|
| 0 | 0.800 | 0.000 | 0.8 | 0.8 |
| 1 | 0.500 | 0.283 | 0.1 | 0.7 |
| 2 | 0.167 | 0.125 | 0.0 | 0.3 |
| 3 | 0.233 | 0.205 | 0.0 | 0.5 |
| 4 | 0.300 | 0.216 | 0.0 | 0.5 |
| 5 | 0.233 | 0.205 | 0.0 | 0.5 |
| 6 | 0.300 | 0.294 | 0.0 | 0.7 |
| 7 | 0.767 | 0.125 | 0.6 | 0.9 |
| 8 | 0.200 | 0.283 | 0.0 | 0.6 |
| 9 | 0.333 | 0.287 | 0.0 | 0.7 |

Solver calls per seed: 5,000 free + 5,000 nudged + 100 evaluation = 10,100.

### Observations

- Digit 2 is the weakest class (0.17). Digits 0 and 7 are the strongest (0.80
  and 0.77) and the only ones every seed scored above 0.5.
- Class 6 has the largest spread (std 0.29). Classes 1, 8 and 9 are close
  behind, so accuracy on them depends strongly on the seed.
- The fraction of samples whose hidden state changed starts between 0.69 and
  0.93 and falls over the five epochs, to 0.23 on seed 0 and about 0.52 on the
  other two. Every epoch stays above the 0.05 learning threshold.
- Tie rate tracks accuracy inversely: seed 0 is both the weakest (0.15) and the
  tiediest (0.62 against 0.06 and 0.18), so on most of its test images the
  output blocks scored equally and the prediction came from the local-field
  fallback rather than a decisive output state.

### Recorded baselines

| Run | Accuracy |
|---|---|
| Paper-style software annealer | 0.15 |
| Trivial gate | 0.10 |
| Backprop MLP, same split | 0.91 |
| **This experiment** | **0.3833** |

Above the annealer baselines, below backprop. Network, solver, and data scale
are in the [project limitations](../../README.md#limitations).
