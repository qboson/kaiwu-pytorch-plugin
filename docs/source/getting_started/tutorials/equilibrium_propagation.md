# Equilibrium propagation on MNIST

## Training setup

This example trains a 784-128-10 network with EPLinear layers and discrete spins. Kaiwu simulated annealing solves the free and nudged phases; PyTorch differentiates their energy difference to update the parameters.

Use Python 3.10.11, Kaiwu 1.3.1, PyTorch 2.7.0 and torchvision 0.22.0. Initialize the local SDK license if required. Run the following command from the repository root, using a fresh output directory and an existing MNIST data directory.

```bash
python example/ep_mnist/train_mnist.py --epochs 3 --seed 0 --split-seed 0 --data-dir data --output-dir runs/ep-mnist-final
```

The official MNIST dataset contains 60,000 training and 10,000 test images. The project reserves 5,000 of the training images for validation, leaving 55,000 for training. Training and split seeds are both 0. SGD uses learning rate 0.05 and batch size 20; beta is 2 and the first-layer initialization scale is 2. The annealer uses initial temperature 10, alpha 0.9, cutoff 0.01, 50 iterations per temperature and four candidates.

## Completed results

After three epochs, EP reaches 90.06% training, 89.54% validation and 90.30% test accuracy. The test set is evaluated only after the final epoch. These are single-seed results. The historical one-epoch linear result is 88.72%; the historical one-epoch convolutional result remains 71.55%.

The BP comparison reaches 96.20% test accuracy with the same split, seed, layer widths, batch size, learning rate and epoch count. BP uses ReLU, cross entropy and default PyTorch initialization, whereas EP uses discrete spins and its own initialization. The accuracy gap is 5.90 percentage points.

## Training curves and interpretation

![MNIST classification error](../../../_static/ep-mnist/training_curves.png)

The curves show training and validation classification error, equal to one minus accuracy. No per-epoch test curve is available. The EP energy-difference objective can be negative and is not comparable in magnitude to BP cross entropy.

EP averages 10,422.09 seconds per training epoch and 5,653.59 seconds for train/validation evaluation; BP averages 5.16 and 2.72 seconds. Final test time is separate. These measured wall times describe the recorded environment, not a general performance guarantee.

The beta and annealing-budget comparisons use 2,000 training and 1,000 validation images, three epochs and seeds 0/1/2. They are validation-only experiments, not additional full-MNIST test results.

[PDF](../../../_static/ep-mnist/training_curves.pdf) · [CSV](../../../_static/ep-mnist/training_curves.csv)
