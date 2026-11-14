# Ising Equilibrium Propagation on Kaiwu

[English](README.md) · [中文](README.zh-CN.md)

Equilibrium propagation (EP) trains an energy-based network using only two
relaxations of that network, with no backward pass. This project implements EP
over **discrete Ising spins** and delegates each relaxation to the
[Kaiwu](https://platform.qboson.com/) SDK's simulated annealer, so the
"relaxation" step is a real combinatorial solve rather than a gradient descent.

The main experiment classifies sklearn Digits with a 64-64-40 network. The
default configuration reaches **10.0% ± 0.2%** test accuracy over three seeds
(tie rate 100%); halving the output-layer learning rate, chosen on a validation
split, reaches **49.9% ± 1.3%**. A backpropagation baseline on the same split
reaches 93.52% ± 0.65%. The archive is in
[`results/digits/`](results/digits/tuned/README.md) and the numbers are
reproduced in [Results](#results) below. MNIST/100 remains as a second
experiment in [`results/primary/`](results/primary).

---

## Algorithm

### Network and encoding

Inputs `x` are clamped pixel values. Hidden spins `h` and output spins `o` are
bipolar, `s ∈ {-1, +1}`. Each of the 10 classes owns a block of 4 output spins,
so `n_output = 40`. The target for class `c` sets that block to `+1` and all
other spins to `-1`.

### Energy and cost

```
E(x, h, o) = -x·W·h - h·V·o - b_h·h - b_o·o
C(o, y)    = n_output - y·o            # equals 0.5*||o - y||^2 for ±1 spins
F_β        = E + β·C
```

The squared output error collapses to a *linear* term in `o`, which is what
keeps the whole objective quadratic and therefore solvable as a QUBO.

### Two phases

Collecting the input and bias terms into local fields

```
H_h = x·W + b_h
H_o = b_o + β·y
```

turns the objective into a problem over spins alone:

```
F_β(h, o) = -h·V·o - H_h·h - H_o·o + β·n_output
```

The **free phase** solves this at `β = 0`, so no label information enters. The
**nudged phase** solves it at `β > 0`, where the label appears only through
`H_o`. The nudged solve warm-starts from the free phase's solution, which is this
project's stand-in for the reverse annealing used on quantum hardware: it keeps
the two states close so their difference reflects the nudge rather than solver
noise.

### Gradient

EP's estimator is the difference between the two phases' correlations. Here it is
obtained by autograd on

```
loss = (E(x, h_nudged, o_nudged) - E(x, h_free, o_free)) / β
```

Because the spins are constants returned by the solver, differentiating this
expression yields exactly

```
dW   = (x ⊗ h_free - x ⊗ h_nudged) / β
dV   = (h_free ⊗ o_free - h_nudged ⊗ o_nudged) / β
db_h = (h_free - h_nudged) / β
db_o = (o_free - o_nudged) / β
```

so PyTorch replaces the hand-written formula. Every sample in a batch is solved
against the *same* parameters; one optimizer step is taken per batch.

### QUBO mapping

Kaiwu consumes an upper-triangular QUBO. Substituting `s = 2q - 1` with
`q ∈ {0, 1}`, and using `q² = q` to fold all linear terms onto the diagonal:

```
Q[:n_h, :n_h] = diag(2·V.sum(axis=1) - 2·H_h)
Q[n_h:, n_h:] = diag(2·V.sum(axis=0) - 2·H_o)
Q[:n_h, n_h:] = -4·V
constant      = β·n_output + H_h.sum() + H_o.sum() - V.sum()
```

`constant` shifts the objective but not its minimizers, so it is returned
alongside `Q` rather than folded in.

### Deeper networks

The model is a stack of layers: `--hidden-sizes 128,64` gives 784-128-64-40.
Layer `l` has its own weight `W_l` and bias `b_l`, and the energy sums over
adjacent layers only:

```
E = -Σ_l s_{l-1}·W_l·s_l - Σ_l b_l·s_l        (s_0 = x, the last s is the output)
```

`IsingEPModel.qubo(x, target, β)` folds this into one block-tridiagonal
`Q`. Each coupling adds `-4·W` to the block between its two layers and
`2·W.sum(axis)` to each diagonal block; each field adds `-2·H`. The single-hidden
case above is the hidden-to-output coupling `V` and the fields `H_h`, `H_o`.
Learning rates are set per layer (`--lr-weights`, `--lr-biases`, one value per
connection), with
optional `--weight-decay` on weights only and `--init-gains` to scale the initial
uniform range of each layer.

The default `--n-hidden 64` builds exactly the network and initial weights of the
archived run.

### Auxiliary spin

The SDK's QUBO-to-Ising conversion appends one gauge spin (an Ising coupling
matrix has no linear terms, so the linear part needs a reference spin). Solver
rows therefore have `n + 1` entries. After ranking the candidates by the Ising
objective and applying the negative-tail flip, that gauge spin is fixed at `+1`.
Both gauge choices become the same physical state, which is the leading `n`
entries. The Ising objective differs from the EP energy by a constant, so they
share a minimizer. The seeded anneal fixes the candidates, so runs are
bit-for-bit reproducible.

`kaiwu_solver.py` selects the SDK entry points at call time, so one code path
serves Kaiwu 1.3.1 to 1.4.3. On 1.4.x the helpers live on `kaiwu.core`
(`qubo_model_to_ising_model`, `get_sorted_solutions`). On 1.3.1 they are
`kaiwu.qubo.qubo_matrix_to_qubo_model` and `kaiwu.conversion`, and the ranking
and negative-tail flip are done locally with the same rule.

---

## Layout

```
src/ising_ep/            Reusable core, no experiment specifics
  model.py               Layer stack, parameters, EP energy and QUBO (torch.nn.Module)
  kaiwu_solver.py        The only module that imports kaiwu
examples/mnist_kaiwu/    The MNIST/100 experiment
  data.py                Deterministic MNIST/100 subset
  train.py               Two-phase training loop
  evaluate.py            Free-phase-only evaluation
  metrics.py             Accuracy, dynamics and learning evidence
  artifacts.py           JSON/NPZ writing and optional credentials
  experiment.py          CLI entry point
examples/digits_kaiwu/   sklearn Digits experiment and a backprop baseline
tests/unit/              No SDK or credentials needed
tests/integration/       Skipped automatically without the licensed SDK
results/digits/          Digits main experiment: reference and tuned runs
results/primary/         MNIST/100 three-seed run (read-only evidence)
results/tuned/           MNIST/100 tuned runs
```

Only `kaiwu_solver.py` touches the SDK, and it imports it lazily, so everything
else is importable and testable without the enterprise wheel.

---

## Setup

Supported Kaiwu versions are 1.3.1 to 1.4.3 (`pyproject.toml` lists
`kaiwu>=1.3.1,<=1.4.3` only in the optional `kaiwu` extra, because the wheel is
not on PyPI; install the wheel by hand as described below, and `pip install -e`
succeeds without it). The archived result was
produced on 1.4.1. The 1.4 line asks for a user id and SDK code before the first
anneal (see `.env.example`); 1.3.1 does not. The wheels target **Python 3.10**.

The integration tests have run against the real 1.3.1 and 1.4.1 wheels; 1.4.2
and 1.4.3 are inside the declared range but have not been run. The 1.4 wheel
gets `kaiwu/__init__.py` and `kaiwu.core` from its `kaiwu-community` dependency.
Replacing 1.3.1 with 1.4.x in the same environment deletes those shared files,
so run `pip install --force-reinstall --no-deps kaiwu-community==1.0.7`
afterwards, or start from a fresh environment.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If you would rather not install the package, `requirements.txt` pins the same
runtime and development dependencies (numpy, torch, pytest, pylint, pydocstyle):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

It does not include the Kaiwu wheel, which is installed separately.

<details>
<summary>Installing without network access</summary>

`pip install -e .` builds in an isolated environment by default, which means
fetching `setuptools>=68` from PyPI. Offline, reuse the interpreter's own build
tools instead:

```powershell
.\.venv\Scripts\python.exe -m pip install wheel
.\.venv\Scripts\python.exe -m pip install -e . --no-build-isolation --no-deps
```

`wheel` is needed because setuptools below 70 has no built-in `bdist_wheel`
command; without it the build stops at `invalid command 'bdist_wheel'`. With
`--no-deps`, make sure `numpy`, `torch` and the dev tools are already installed.

If you cannot install at all, the test suite still runs: `pyproject.toml` sets
`pythonpath = [".", "src"]`, so `pytest` resolves `ising_ep` in a bare checkout.
Running the experiment module directly does need the package importable, so
either install it or set `PYTHONPATH=src`.

</details>

The Kaiwu wheel is not on PyPI. Download the `cp310` wheel for your OS and
architecture from the quantum cloud platform and install it into the same
interpreter:

```powershell
.\.venv\Scripts\python.exe -m pip install <kaiwu-1.4.1-cp310-...>.whl
.\.venv\Scripts\python.exe -c "import importlib.metadata as m; print(m.version('kaiwu'))"
```

That last command prints the installed version. On 1.4.x `kaiwu.__version__`
reports a different, stale value; the distribution version is the authoritative
one.

The 1.4 line asks for credentials before the first anneal: copy `.env.example`
to `.env` and fill in `KAIWU_USER_ID` and `KAIWU_SDK_CODE`. Version 1.3.1 does
not. Lookup order: environment variables first, then `.env` in the project
directory, then `.env` in its parent (the workspace root). Never commit `.env`
or a license file.

### MNIST data

The data is not vendored. Either place the four IDX archives in
`data/MNIST/raw/`, or pass `--download-data` to fetch them on first use.
Downloading is opt-in so a training run never reaches the network unexpectedly.

---

## Running

Digits is the main experiment. The command that produced
[`results/digits/tuned/lrout05/`](results/digits/tuned/lrout05):

```powershell
.\.venv\Scripts\python.exe -m examples.digits_kaiwu.experiment `
  --output runs/digits-lrout05 --seeds 0,1,2 --epochs 5 `
  --validation-fraction 0.2 --eval-every-epoch `
  --lr-weights 0.01,0.005 --lr-biases 0.01,0.005
```

The backpropagation baseline on the same split does not need Kaiwu:

```powershell
.\.venv\Scripts\python.exe -m examples.digits_kaiwu.backprop `
  --output runs/digits-backprop --seeds 0,1,2
```

MNIST/100 is the second experiment. Smoke run (one seed, small network, minutes):

```powershell
.\.venv\Scripts\python.exe -m examples.mnist_kaiwu.experiment `
  --output runs/smoke --seeds 0 --n-hidden 8 --epochs 1 `
  --candidate-count 2 --iterations-per-t 4
```

The full three-seed run that produced `results/primary/`:

```powershell
.\.venv\Scripts\python.exe -m examples.mnist_kaiwu.experiment `
  --output runs/primary --seeds 0,1,2 `
  --n-hidden 64 --epochs 5 --batch-size 10 --lr 0.01 --beta 1.0
```

A deeper network with per-layer learning rates (three connections, so three
values each):

```powershell
.\.venv\Scripts\python.exe -m examples.mnist_kaiwu.experiment `
  --output runs/deep --seeds 0 --hidden-sizes 128,64 `
  --lr-weights 0.01,0.01,0.02 --lr-biases 0.005,0.005,0.01
```

No accuracy is recorded for deeper networks on MNIST. On Digits,
`--hidden-sizes 128` and `128,64` did not beat the reference during screening;
see [`results/digits/tuned/`](results/digits/tuned/README.md).

`--help` lists every option, including the annealing schedule
(`--initial-temperature`, `--alpha`, `--cutoff-temperature`,
`--iterations-per-t`, `--candidate-count`).

### Output

```
<output>/config.json              Full configuration, for reproduction
<output>/summary.json             Cross-seed statistics
<output>/seeds/<seed>/
    initial-model.npz             Parameters before training
    checkpoint.npz                Parameters after the final epoch
    epochs/<e>/model.npz          Per-epoch snapshot
    epochs/<e>/dynamics.json      Flip rates for that epoch
    test/metrics.json             Accuracy, tie rate, confusion, per class
    test/predictions.json         Per-sample predictions
    status.json                   Seed summary and learning evidence
```

The `.npz` files store `weights.0 … weights.K` and `biases.0 … biases.K`, input
layer first. The archived run predates this layout and uses the single-hidden
names: `W`, `V`, `b_h`, `b_o` correspond to `weights.0`, `weights.1`, `biases.0`,
`biases.1`, with identical values.

What is committed under `results/` is `config.json`, `summary.json`, each seed's
`status.json` and `test/metrics.json`, plus `epochs/<e>/validation.json` when a
run used a validation split. Parameter snapshots, flip rates and per-sample
predictions stay in the local `runs/` directory.

Everything random derives from the seed, so a given `(seed, configuration)` pair
reproduces bit for bit: initialization, the per-epoch shuffle, each sample's
paired anneal seed, and the evaluation seeds.

---

## Results

### Digits

Archived in [`results/digits/`](results/digits/tuned/README.md). Network
64-64-40, 5 epochs, batch 10, seeds 0/1/2, split seed 0 (1,149 train / 288
validation / 360 test). Annealing matches MNIST: `T₀ = 1.25`, `α = 0.95`,
cutoff `0.001`, 10 iterations per temperature, 8 candidates per solve.
Hyperparameters were chosen on the validation split, and each seed reads the
test split once.

| Configuration | Test accuracy | Tie rate |
|---|---|---|
| Default (`lr = 0.01`, `β = 1`) | 10.0% ± 0.2% | 100% |
| Output learning rate ×0.5 | **49.9% ± 1.3%** | 6.6% |
| Plus `β = 2` | 49.3% ± 0.6% | 5.0% |
| Backprop 64-64-10 (`lr = 0.05`) | 93.52% ± 0.65% | |

The standard deviation is the population standard deviation of the three seeds.
The default configuration ties on every test image. The two tuned
configurations differ by 0.6 points, less than the spread across seeds. The
backprop network uses the same split, epoch count and batch size, with its own
default learning rate.

### MNIST/100

Archived in [`results/primary/`](results/primary). Configuration: 784-64-40,
5 epochs, batch 10, `lr = 0.01`, `β = 1.0`, seeds 0/1/2; annealing
`T₀ = 1.25`, `α = 0.95`, cutoff `0.001`, 10 iterations per temperature,
8 candidates per solve.

A follow-up tuning study, with hyperparameters chosen on a validation split and
the test set used once, is in [`results/tuned/`](results/tuned): initial output-layer
gain 0.5 reaches 62.7% ± 1.9% test accuracy. The numbers above are unchanged.

Each seed makes **10,100 solver calls**: 5,000 free + 5,000 nudged during
training (1,000 samples × 5 epochs × 2 phases), plus 100 for evaluation.

| Seed | Accuracy | Tie rate | ‖ΔW‖ | ‖ΔV‖ |
|------|----------|----------|------|------|
| 0 | 0.15 | 0.62 | 2.49 | 2.66 |
| 1 | 0.54 | 0.06 | 3.50 | 3.35 |
| 2 | 0.46 | 0.18 | 3.00 | 3.05 |
| **mean** | **0.3833** | 0.2867 | | |
| std | 0.1682 | 0.2407 | | |

The worst seed is also the one that tied most often: seed 0 fell back to the
local-field tie-break on 62% of test images, against 6% and 18% for the other
two.

Per-class accuracy (mean over seeds):

| Class | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|-------|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 0.80 | 0.50 | 0.17 | 0.23 | 0.30 | 0.23 | 0.30 | 0.77 | 0.20 | 0.33 |

Digit 2 is the weakest class at 0.17; digits 0 and 7 are the strongest. The
share of samples whose hidden state changed between the two phases runs from
0.23 to 0.93 depending on the seed and the epoch, and every epoch stays above
0.05, so the nudge did not collapse to the free state.

Recorded baselines on the same split: a paper-style software annealer 0.15, a
trivial gate 0.10, and a backprop MLP 0.91. See [Limitations](#limitations).

---

## Tests and checks

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m pytest tests -m integration
.\.venv\Scripts\python.exe -m pylint src/ising_ep examples tests
.\.venv\Scripts\python.exe -m pydocstyle --convention=google src examples
```

Unit tests need neither the SDK nor credentials: a fake `kaiwu` module is
injected to assert the official call chain. Integration tests skip automatically
when the wheel is absent.

Docstrings are enforced, not just encouraged: `pylint.extensions.docparams` is
enabled with `accept-no-param-doc = false`, so a missing `Args`, `Returns` or
`Raises` section fails the lint.

---

## Limitations

- **Network.** The archived run uses 64 hidden spins; the literature uses 120.
  Deeper stacks are supported but not benchmarked here.
- **Solver.** Kaiwu's classical simulated annealer with a warm start. The
  literature uses D-Wave reverse annealing.
- **Data.** The main experiment is sklearn Digits: 1,149 training images, 288
  validation images and 360 test images. MNIST/100 is still here: 1,000
  training images and 100 test images.
- **Accuracy.** On Digits, tuned EP reaches **49.9% ± 1.3%** and the same-split
  backprop baseline reaches 93.52% ± 0.65%; the default EP configuration is
  10.0% ± 0.2%. The MNIST/100 reference run is **38.3% ± 16.8%**. Same-split
  baselines there: software annealer 0.15, trivial gate 0.10, backprop MLP 0.91.

## Sources

Papers are not included in this repository.

- Laydevant, Marković, and Grollier, *Training an Ising machine with equilibrium
  propagation* (2024). Discrete Ising EP, ±1 spins, one block of output spins
  per class.
- Scellier and Bengio, *Equilibrium Propagation: Bridging the Gap between
  Energy-Based Models and Backpropagation* (2017). The ``F_β`` objective and
  the two-phase gradient.
- Ernoult et al., *Equilibrium Propagation with Continual Weight Updates*
  (2020). Laborieux et al., *Scaling Equilibrium Propagation to Deep ConvNets
  by Drastically Reducing Its Gradient Estimator Bias* (2021). Scellier et al.,
  *Quantum Equilibrium Propagation* (2024).
