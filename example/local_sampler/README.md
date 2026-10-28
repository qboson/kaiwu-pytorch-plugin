# Bring Your Own Sampler

This example shows the **sampler contract** that `Kaiwu-PyTorch-Plugin` relies on
and verifies a custom solver against brute force, so the contract can be
exercised offline without Kaiwu SDK credentials.

## What the plugin expects from `solve()`

`model.get_ising_matrix()` returns an `(N + 1) x (N + 1)` auxiliary-spin matrix
`M`. The model energy satisfies `E(x) = c - y^T M y` with `y = [2x - 1, +1]`, so
the low-energy (high-probability) states are the **maximizers** of `s^T M s` --
this is the direction the Kaiwu SDK solvers use. A sampler must therefore:

1. accept the NumPy matrix and return an array shaped `(num_reads, N + 1)`,
2. emit spins in `{-1, +1}` only,
3. return states that maximize `s^T M s` (equivalently: minimize the model energy).

The plugin decodes a returned row with `spins[:-1] * spins[-1]` and maps the
result back to the model's `0/1` state.

## Contents

| File | Purpose |
| --- | --- |
| `local_sampler.py` | `MetropolisSampler` (pure PyTorch, simulated annealing) plus the verification and training smoke test |
| `README.md` / `README_ZH.md` | This document and its Chinese translation |

## Run

```bash
python example/local_sampler/local_sampler.py
```

The script builds a four-node Boltzmann machine with three degenerate ground
states, enumerates all `2^4` states, samples through `model.sample(sampler)` and
compares the decoded results with the exhaustive minimum:

```text
ground states (exhaustive): [(0, 0, 1, 1), (0, 1, 1, 1), (1, 0, 1, 1)]
ground energy             : -1.5000
decoded samples           : [(0, 1, 1, 1), (1, 0, 1, 1), ...]
minimal-energy draws      : 32/32 -> PASS
training smoke test       : objective 0.0377 -> 0.1591 over 5 step(s)
```

`--steps` and `--reads` control the training smoke test and the number of
samples per `solve()` call; `--seed` makes the run reproducible.

## Using it in your own loop

```python
from local_sampler import MetropolisSampler

sampler = MetropolisSampler(num_reads=8, num_sweeps=100, seed=0)
negative = rbm.sample(sampler)          # decoded 0/1 states
```

Because the sampler interface is duck-typed, the same object works with
`RestrictedBoltzmannMachine`, `BoltzmannMachine` and `GBRBM`; only `solve()`
matters.

## Dependencies

Only the project requirements (`torch`, `numpy`) -- no Kaiwu license is needed.
