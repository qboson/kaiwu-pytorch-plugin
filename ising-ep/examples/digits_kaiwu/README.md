# Ising EP on sklearn Digits

[English](README.md) · [中文](README.zh-CN.md)

Trains a 64-64-40 Ising equilibrium-propagation network on sklearn Digits, with
both relaxation phases solved by Kaiwu's simulated annealer. A backpropagation
baseline uses the same split. The algorithm is in the
[project README](../../README.md).

## Data

`sklearn.datasets.load_digits`: 1,797 images of 8×8 digits, pixels divided by
16 into `[0, 1]`. The split is stratified and `--split-seed` (default 0) is
independent of the experiment seeds: 360 images (20%) are the test set and the
other 1,437 are for training. `--validation-fraction 0.2` then holds 288 of
those training images out for validation and does not touch the test set.

Nothing is downloaded. The `data` block of `config.json` records the split
seed, the fractions and the sizes, so the partition can be rebuilt from that
file alone.

## Commands

Run from the repository root with `ising_ep` importable (`pip install -e
".[dev]"`, or `PYTHONPATH=src;.`).

Equilibrium propagation, default configuration:

```powershell
python -m examples.digits_kaiwu.experiment --output runs/digits-primary --seeds 0,1,2
```

Choose hyperparameters on the validation split without reading the test split:

```powershell
python -m examples.digits_kaiwu.experiment --output runs/digits-val `
  --validation-fraction 0.2 --eval-every-epoch --skip-test --seeds 0 `
  --lr-weights 0.01,0.005 --lr-biases 0.01,0.005
```

The backpropagation baseline does not need Kaiwu. It is a 64-64-10 network
(tanh hidden layer, cross-entropy) with the same epoch count, batch size and
split as the EP run, and its own default learning rate of 0.05:

```powershell
python -m examples.digits_kaiwu.backprop --output runs/digits-backprop --seeds 0,1,2
```

The QUBO holds only the hidden and output spins, so a Digits solve costs the
same as a MNIST solve of the same network.

## Archived results

| Run | Test accuracy | Where |
|---|---|---|
| EP, default | 10.0% ± 0.2% | [`results/digits/primary/`](../../results/digits/primary/README.md), tie rate 100% |
| EP, output learning rate ×0.5 | **49.9% ± 1.3%** | [`results/digits/tuned/`](../../results/digits/tuned/README.md) |
| EP, plus `β = 2` | 49.3% ± 0.6% | same |
| Backpropagation | 93.52% ± 0.65% | Default settings, 3 seeds, about 0.04 s per epoch |

The screening log and the reproduction commands are in the tuned-results note.
