# Equilibrium propagation on MNIST

## Installation and result provenance

Use Python 3.10 and an installed Kaiwu 1.3.1 SDK. From the repository root:

```bash
python -m pip install -e .
python -m pip install -r example/ep_mnist/requirements.txt
```

The example dependencies add torchvision 0.22.0 and matplotlib 3.10.3; they
do not change the plugin's runtime requirements. The default is one epoch;
pass `--epochs 3` for the historical full-data configuration. `--evaluate`
restores a checkpoint for evaluation, not for resuming training.

The three-epoch results below are historical runs from `93a536b` plus local
changes, identified by the source hashes in [provenance.json](closeout/provenance.json).
They were not rerun in full for this final submission. New small-run checks
are documented separately in [validation](../../tests/README.validation.md).

This example supports full-data training and a quick check on a small MNIST
subset. Use `--limit-train 20 --limit-eval 20` with a separate `--output-dir`
for the quick check; see the command below.

Trains equilibrium-propagation networks on the full original MNIST, read with
`torchvision.datasets.MNIST`. Pixels are divided by 255. Every unit is a ±1
spin, and the Ising matrices of the free and nudged phases are relaxed by Kaiwu
1.3.1 simulated annealing.

The test split is the official 10000 test images. 5000 of the 60000 training
images are held out at random for validation, and the other 55000 are used for
training. `--split-seed` (default 0) fixes the split independently of `--seed`
(default 0), which controls initialization, training order, and training anneals.
Hyperparameters were chosen on validation accuracy with 2000 training images.

The example wraps tensors in a `TensorDataset` and feeds batches through a
`DataLoader`. Build `EquilibriumPropagation` from `EPLinear` and `EPConv2d`
layers. For each batch, call `equilibrate(inputs, sampler)` for the free phase,
and `equilibrate(inputs, sampler, targets, beta)` for the nudged phase. Labels
use +1 for their class and -1 elsewhere. Backpropagate `objective(inputs, free,
nudged, beta)` to update the parameters with a PyTorch optimizer. Prediction
takes the argmax of the last layer's fields computed from the free-phase state.

The clamped pixels and biases contribute to the local fields. The nudge adds
`beta * target` to the output field. The reference spin encodes local fields
in the Ising matrix, and decoding selects the minimum-energy returned candidate.
Both phases use the same annealer settings. `--sa-iterations` sets the annealing
iterations per temperature. States use discrete flips and have no continuous
step size.

## Fully connected network (default)

784-128-10, first-layer init scaled by 2, beta 2, SGD learning rate 0.05, batch
size 20. Annealing runs 50 iterations per temperature: with 10, single spin
flips still lowered the energy, so the solutions had not converged.

```powershell
python example/ep_mnist/train_mnist.py --epochs 3 --output-dir runs/ep-mnist-linear
```

The first run downloads MNIST to `example/ep_mnist/data/`. Pass `--data-dir` to
use an existing copy. The completed three-epoch result is recorded below.

## Final three-epoch result

The completed full-data run uses seed 0 and split seed 0: 55,000 training,
5,000 validation and 10,000 official test images. The final training accuracy
is **90.06%**, validation accuracy **89.54%**, and test accuracy **90.30%**.
The test set was evaluated only after epoch 3. This is one seed, not a
multi-seed mean. The historical one-epoch test result was 88.72%.

The BP comparison reaches **96.20%** test accuracy on the same split, seed,
layer widths, batch size, learning rate and three epochs. It uses ReLU,
cross entropy and PyTorch's default initialization; EP uses discrete spins
and its own initialization. The gap is 5.90 percentage points.

EP averages 10,422.09 seconds per training epoch and 5,653.59 seconds per
train/validation evaluation pass. BP averages 5.16 and 2.72 seconds,
respectively. These are measured wall times on the recorded environment,
not a general speed benchmark. Final test evaluation time is separate.

![Three-epoch training and validation classification error](../../docs/_static/ep-mnist/training_curves.png)

[Curve PDF](../../docs/_static/ep-mnist/training_curves.pdf) ·
[Per-epoch CSV](../../docs/_static/ep-mnist/training_curves.csv) ·
[EP summary](closeout/ep-full/summary.json) · [BP summary](closeout/bp/summary.json)

The curve shows classification error (1 - accuracy), not test accuracy.
EP's two-phase energy-difference objective can be negative and is not BP
cross entropy; compare classification error rather than objective magnitude.
The beta/annealing study uses 2,000 training and 1,000 validation images,
three epochs and seeds 0/1/2; its results are validation-only.
See the [evidence archive](closeout/README.md) for configuration, provenance
and figure regeneration. The convolutional result below remains a historical
one-epoch run; no new three-epoch convolutional result is claimed.

## Convolutional network

`--conv` replaces the first layer with an `EPConv2d`: 4 channels, kernel 7,
stride 3, giving 4×8×8 = 256 spins, followed by 10 output spins. With more spins,
annealing runs 100 iterations per temperature, and the first-layer init is
scaled by 4.

```powershell
python example/ep_mnist/train_mnist.py --conv --input-scale 4 --sa-iterations 100 --output-dir runs/ep-mnist-conv
```

Recorded results after one epoch with seed 0:

Validation accuracy: 0.6966. Test accuracy: 0.7155.

## Run time

Each training sample calls the annealer once per phase, so the update pass takes
110000 solves per epoch. Train and validation evaluation take another 60000
free-phase solves per epoch. The final test uses 10000 solves. The completed three-epoch run averages
2.90 hours for training and 1.57 hours for train/validation evaluation per epoch
(4.47 hours combined); final test evaluation is separate. Earlier measurements
of about 31 ms per solve (roughly one hour for the update pass), and 72/174 ms
per solve for parallel linear/convolutional runs, are historical measurements
and do not estimate the completed three-epoch run. Before a
full run, `--limit-train 2000 --limit-eval 1000` shows the speed on your machine.

Use Python 3.10, Kaiwu 1.3.1, PyTorch 2.7.0 and torchvision 0.22.0.
Initialize the local SDK license before running if your installation requires it.

The default output is `example/ep_mnist/results/linear/` or `results/conv/`.
Use a new directory for every run. After each epoch the script updates
`checkpoint.pt` and `summary.json` (`summary_conv.json` for convolution), with
train and validation accuracy, classification error, EP objective, and separate
training and evaluation times. `--validation-only` skips the test set for tuning.

```powershell
python example/ep_mnist/train_mnist.py --limit-train 20 --limit-eval 20 --output-dir runs/ep-mnist-smoke
python example/ep_mnist/train_mnist.py --evaluate runs/ep-mnist-smoke/checkpoint.pt --output-dir runs/ep-mnist-smoke-eval
```

The evaluation command rebuilds the saved architecture, restores its PyTorch
`state_dict`, and uses the stored split and evaluation sampler configuration.
If training used `--data-dir`, pass it again when evaluating. The result is
written to `evaluation.json`.
