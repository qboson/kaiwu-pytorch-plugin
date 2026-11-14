# Tuned runs

Two configurations that beat the reference run on a validation split, then were
scored on the 100 test images **exactly once**. The reference is
[`results/primary/`](../primary) (784-64-40, `β = 1`, 38.3% ± 16.8% test accuracy).

The archive keeps the configuration, the summary, each seed's status, the test
metrics and the per-epoch validation records. Parameter snapshots and
per-sample predictions stay local.

## Protocol

- **Hyperparameters were chosen on validation only.** The validation split is
  100 images, 10 per class: images 101 to 110 of each class in the MNIST
  training file, disjoint from the 1,000 training images (`--validation-per-class 10`).
- The test split was never read while tuning (`--skip-test`). Only the two final
  configurations below were evaluated on it, once per seed, after 5 epochs.
- Everything ran on Kaiwu 1.4.1, the same version as the reference.
- Screening used seed 0, 3 epochs, one change at a time from the reference. The
  best candidates were re-checked on seeds 1 and 2, and a combination was added.

## Screening (validation accuracy after 3 epochs)

Seed 0 only, reference 17% with 59% ties:

| Change | Val acc | Tie rate |
|---|---|---|
| `--init-gains 1,0.5` | 60% | 5% |
| `--beta 2` | 56% | 9% |
| output-layer learning rates x0.5 | 50% | 19% |
| `--iterations-per-t 20` | 34% | 2% |
| `--candidate-count 16` | 17% | 59% |
| `--weight-decay 1e-4` | 18% | 63% |
| `--iterations-per-t 40` | 10% | 100% (all predicted class 3) |
| `--beta 0.5` | 10% | 0% (all predicted class 7) |
| output-layer learning rates x2 | 10% | 100% (all one class) |
| `--init-gains 1,2` | 10% | 100% (all one class) |
| `--hidden-sizes 128` | 10% | 98% (all one class) |
| `--hidden-sizes 128,64` | 9% | 78% (almost all class 9) |

The failed settings predict a single class for every image, so their 10% is
chance. Deeper or wider networks did not help at this training length.
Every candidate was a one-factor change run in parallel rather than a sequential
greedy search, and "output-layer x2 / x0.5" scales both weight and bias rates.

## Re-check on seeds 0, 1, 2 (validation accuracy after 3 epochs)

| Config | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| reference | 17% | 50% | 32% | 33.0% |
| `--init-gains 1,0.5` | 60% | 51% | 63% | 58.0% |
| `--beta 2` | 56% | 53% | 49% | 52.7% |
| both | 54% | 57% | 56% | 55.7% |

All three are above the reference on every seed, though seed 1 for
`--init-gains 1,0.5` is only 1 point ahead. The reference is mostly weak because
of seed 0 and seed 2 tie-heavy epochs, which the changes remove.

## Final results (5 epochs, test set, once)

| Config | Directory | Test acc (mean ± std) | Per seed | Tie rate (mean) |
|---|---|---|---|---|
| `--init-gains 1,0.5` | [`gain05/`](gain05) | **62.7% ± 1.9%** | 60 / 64 / 64 | 6.7% |
| `--init-gains 1,0.5 --beta 2` | [`gain05-beta2/`](gain05-beta2) | **62.3% ± 6.6%** | 71 / 61 / 55 | 5.0% |
| reference (`results/primary/`) | [`../primary/`](../primary) | 38.3% ± 16.8% | 15 / 54 / 46 | 28.7% |

Standard deviation is the population standard deviation over three seeds. Final
validation accuracy after epoch 5: 57.0% for `gain05`, 55.3% for `gain05-beta2`.
Each seed makes 10,100 training solves plus 100 test and 500 validation solves.
Epoch-by-epoch validation curves are in `seeds/<s>/epochs/<e>/validation.json`.

## Reading the result

- Both configurations beat the reference by about 24 points of test accuracy and
  cut the tie rate from about 30% to 5% to 7%. The main mechanism is fewer ties
  and fewer collapses onto one class.
- Between the two there is no meaningful winner: means differ by 0.4 points and
  the gap is far inside the seed spread. `gain05` is the simpler one-change
  setting and has the smaller spread.
- Three seeds and 100 test images are a small sample. One image is one point.
  Treat the margin over the reference as real but the ranking of the two tuned
  runs as unresolved.
- Validation and test curves are not monotone in epochs (for example seed 0 of
  `gain05` goes 39, 52, 60, 43, 55% on validation), so the final epoch is noisy.

## Reproduce

```powershell
python -m examples.mnist_kaiwu.experiment --output results/tuned/gain05 `
  --validation-per-class 10 --eval-every-epoch --init-gains 1,0.5
python -m examples.mnist_kaiwu.experiment --output results/tuned/gain05-beta2 `
  --validation-per-class 10 --eval-every-epoch --init-gains 1,0.5 --beta 2
```
