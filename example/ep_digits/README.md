# Equilibrium propagation on Digits

Trains a 64-32-10 equilibrium-propagation network on
`sklearn.datasets.load_digits`. Pixels are divided by 16. Every unit is a ±1
spin, and the Ising matrices of the free and nudged phases are relaxed by Kaiwu
1.3.1 simulated annealing.

The test split is the stratified 25% of `random_state=0`. A further stratified
20% of the remaining images is held out for validation, and hyperparameters were
chosen on validation accuracy only:

- The first-layer weights are scaled by 4 over the default initialization
  (`--input-scale 4`). Otherwise the field a hidden spin receives from the input
  is as large as the feedback from the output, and the hidden layer does not
  learn input features.
- The nudge uses beta 2, and SGD uses learning rate 0.05, batch size 10, and
  6 epochs.
- Prediction runs the free phase only and picks the class with the largest
  output field.

Smoke run (100 training, 100 validation, 100 test images):

```powershell
python example/ep_digits/train_digits.py --smoke
```

Smoke test accuracy: 0.4700.

Full split (1077 train, 270 validation, 450 test):

```powershell
python example/ep_digits/train_digits.py
```

Validation accuracy: 0.8407. Test accuracy: 0.8156. One epoch takes about
80 seconds in a single process.

Both runs write `example/ep_digits/results/summary.json`, which is gitignored.
