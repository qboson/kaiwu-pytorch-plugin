# Equilibrium propagation on MNIST

Trains equilibrium-propagation networks on the full original MNIST, read with
`torchvision.datasets.MNIST`. Pixels are divided by 255. Every unit is a ±1
spin, and the Ising matrices of the free and nudged phases are relaxed by Kaiwu
1.3.1 simulated annealing.

The test split is the official 10000 test images. 5000 of the 60000 training
images are held out at random for validation, and the other 55000 are used for
training. Hyperparameters were chosen on validation accuracy with 2000 training
images.

## Fully connected network (default)

784-128-10, first-layer init scaled by 2, beta 2, SGD learning rate 0.05, batch
size 20. Annealing runs 50 iterations per temperature: with 10, single spin
flips still lowered the energy, so the solutions had not converged.

```powershell
python example/ep_mnist/train_mnist.py
```

The first run downloads MNIST to `example/ep_mnist/data/`. Pass `--data-dir` to
use an existing copy. Results after one epoch:

Validation accuracy: 0.8788. Test accuracy: 0.8872.

## Convolutional network

`--conv` replaces the first layer with an `EPConv2d`: 4 channels, kernel 7,
stride 3, giving 4×8×8 = 256 spins, followed by 10 output spins. With more spins,
annealing runs 100 iterations per temperature, and the first-layer init is
scaled by 4.

```powershell
python example/ep_mnist/train_mnist.py --conv --input-scale 4 --sa-iterations 100
```

Results after one epoch:

Validation accuracy: 0.6966. Test accuracy: 0.7155.

## Run time

Each sample calls the annealer once per phase, so one epoch takes 110000 solves,
plus 15000 for validation and test. In a single process the fully connected
network takes about 31 ms per solve, about one hour per epoch. The two results
above were run in parallel and measured about 72 and 174 ms per solve. Before a
full run, `--limit-train 2000 --limit-eval 1000` shows the speed on your machine.

Results are written to `example/ep_mnist/results/` (`summary.json` for the
fully connected network, `summary_conv.json` for the convolutional one). That
directory and the data directory are gitignored.
