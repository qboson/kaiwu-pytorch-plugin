# Tuned Digits runs

[English](README.md) · [中文](README.zh-CN.md)

Two configurations that beat the reference on the validation split, then
evaluated **once** on the 360 test images. The reference is
[`../primary/`](../primary/README.md) (64-64-40, `β = 1`, test accuracy
10.0% ± 0.2%, tie rate 100%).

The archive keeps the configuration, the summary, each seed's status, the test
metrics and the per-epoch validation records. Parameter snapshots and
per-sample predictions stay local.

## Procedure

- **Hyperparameters were chosen on the validation split only.**
  `--validation-fraction 0.2` holds out 288 stratified images from the training
  part and leaves 1,149 for training. The split seed is the default, 0, the
  same split as the backpropagation baseline.
- Screening and the recheck both pass `--skip-test`, so they never read the
  test split. Only the two configurations below read it, once per seed, after
  5 epochs.
- Every run used Kaiwu 1.4.1.
- Screening used seed 0 and 3 epochs, changing one thing at a time. The two
  settings above the reference were rechecked on seeds 1 and 2, and the pair
  was also stacked.

## Screening (seed 0, validation accuracy after 3 epochs)

The reference is 10.1% with a tie rate of 0, and it assigns almost every image
to one class:

| Change | Validation accuracy | Tie rate |
|---|---|---|
| Output learning rate ×0.5 | 50.0% | 6.2% |
| `--beta 2` | 41.3% | 5.2% |
| `--init-gains 1,0.5` | 12.2% | 0% |
| `--candidate-count 16` | 10.1% | 0% |
| `--weight-decay 1e-4` | 10.1% | 0% |
| `--iterations-per-t 20` | 10.1% | 0% |
| `--iterations-per-t 40` | 10.1% | 0% |
| `--init-gains 1,2` | 10.1% | 100% |
| `--beta 0.5` | 10.1% | 100% |
| `--hidden-sizes 128` | 10.1% | 100% |
| `--hidden-sizes 128,64` | 9.7% | 85.1% |
| Output learning rate ×2 | 9.7% | 100% |

"Output ×0.5 / ×2" scales that layer's weight and bias learning rates together:
`--lr-weights 0.01,0.005` and `--lr-biases 0.01,0.005`, or `0.02` in place of
`0.005`. `--init-gains 1,0.5` scored 41%, 46%, then 12%: it rose for two epochs
and collapsed to one class on the third, so the final epoch kept it out.

## Recheck on seeds 0, 1, 2 (validation accuracy after 3 epochs)

| Configuration | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| Reference | 10.1% | 13.5% | 10.1% | 11.2% |
| Output learning rate ×0.5 | 50.0% | 42.4% | 45.1% | 45.8% |
| `--beta 2` | 41.3% | 39.9% | 33.0% | 38.1% |
| Both | 42.0% | 49.7% | 46.2% | 45.9% |

Both changes, and their combination, beat the reference on every seed. The
combination and the learning-rate change alone differ by 0.1 points. `--beta 2`
alone also beat the reference on every seed, but its mean is about 8 points
lower, so it was not run on the test split by itself.

## Final results (5 epochs, test split, once)

| Configuration | Directory | Test accuracy (mean ± std) | Seeds | Mean tie rate |
|---|---|---|---|---|
| Output learning rate ×0.5 | [`lrout05/`](lrout05) | **49.9% ± 1.3%** | 50.8 / 50.8 / 48.1 | 6.6% |
| Plus `--beta 2` | [`lrout05-beta2/`](lrout05-beta2) | **49.3% ± 0.6%** | 48.6 / 49.2 / 50.0 | 5.0% |
| Reference | [`../primary/`](../primary) | 10.0% ± 0.2% | 9.7 / 10.3 / 10.0 | 100% |

The standard deviation is the population standard deviation of the three seeds.
Validation accuracy after epoch 5: 48.6% for `lrout05`, 50.8% for
`lrout05-beta2`, and 10.0% for the reference. Each seed makes 11,490 training
solves, plus 360 test solves and 1,440 validation solves.

## Reading this

- Both configurations are about 40 points above the reference, and the tie rate
  falls from 100% to 5–7%. After 5 epochs the reference ties on every test
  image.
- There is no meaningful ranking between them: the means differ by 0.6 points,
  less than the spread of `lrout05` across seeds.
- A backpropagation network on the same split and the same 5 epochs reaches
  **93.52% ± 0.65%**. EP is about 44 points lower. That gap is part of the
  result.
- The validation curves are not monotone. Seed 1 of `lrout05`, for example, is
  29%, 42%, 42%, 55%, 48%, so the last epoch is noisy.

## Reproduction

```powershell
python -m examples.digits_kaiwu.experiment --output results/digits/tuned/lrout05 `
  --validation-fraction 0.2 --eval-every-epoch --seeds 0,1,2 `
  --lr-weights 0.01,0.005 --lr-biases 0.01,0.005
python -m examples.digits_kaiwu.experiment --output results/digits/tuned/lrout05-beta2 `
  --validation-fraction 0.2 --eval-every-epoch --seeds 0,1,2 `
  --lr-weights 0.01,0.005 --lr-biases 0.01,0.005 --beta 2
```
