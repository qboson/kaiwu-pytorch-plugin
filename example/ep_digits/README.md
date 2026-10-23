# Equilibrium propagation on Digits

Trains a 64-32-10 equilibrium-propagation network with Kaiwu 1.3.1 simulated
annealing and a same-shaped backprop MLP on `sklearn.datasets.load_digits`.
Pixels are divided by 16. Each unit uses one bit.

A 1-bit unit stays off until its field exceeds 1/2, so the first-layer weights
are initialized at scale 0.2 and the readout near 0.02, with an output bias of
-0.1. The nudge uses beta 2: at beta 1 the target bit ties the all-zero state
and the two phases often stay equal. Training uses SGD with learning rate 0.2
for 6 epochs.

Smoke run (100 training images, 100 test images, 6 epochs):

```powershell
python example/ep_digits/train_digits.py --smoke
```

Smoke test accuracy: EP 0.4800, backprop 0.6300.

The default command uses the stratified 75/25 split (`random_state=0`), the
same 6 epochs and the same learning rate:

```powershell
python example/ep_digits/train_digits.py
```

Full-split test accuracy (1347 train, 450 test): EP 0.8222, backprop 0.9667.

Both runs write `example/ep_digits/results/summary.json`, which is gitignored.
