# Does the Documented Loop Learn? An Exact Check

The Quick Start loop trains an RBM with

```python
positive = rbm.get_hidden(batch, bernoulli=True)
negative = rbm.sample(sampler)
objective = rbm.objective(positive, negative)
objective.backward()
optimizer.step()
```

`objective()` is documented as a surrogate whose gradient matches the negative
log-likelihood gradient, so a step with a positive learning rate must make the
training patterns **more probable**. This example measures that claim instead of
trusting it: for a three-visible / two-hidden RBM the marginal probability of a
pattern can be computed exactly by enumerating all `2^5` states.

No Kaiwu SDK credentials are needed -- the negative phase is the exhaustive
ground state of the current model (what a perfect sampler would return), so the
run is fully deterministic for a given `--seed`.

## What is measured

| Quantity | Meaning |
| --- | --- |
| `p(pattern)` | exact model probability of a training pattern, averaged over the three patterns |
| energy gap | exact mean energy of all `2^5` states minus the patterns' energy; positive means the patterns sit below a typical state |

## Run

```bash
python example/learning_direction/learning_direction.py
```

Expected output (seed 0):

```text
steps=200 seed=0
mean p(pattern) before: 0.124706
mean p(pattern) after : 0.132095
energy gap before     : -0.0022
energy gap after      : 0.0640
PASS: the documented loop increased the probability of the training patterns
```

The probability rises by about 6% and the energy gap turns positive: after
training the patterns sit below the average state, which is exactly what
"learning" means for an energy-based model.

## Why the check is meaningful

`--steps` and `--seed` control the run, and the helper module exposes
`evaluate_learning(..., flip_sign=True)`, which negates the objective. With the
sign flipped the same measurement collapses:

```text
normal : p 0.124706 -> 0.132095,  gap -0.0022 -> +0.0640
flipped: p 0.124706 -> 0.000000,  gap -0.0022 -> +0.4709
```

so the reported PASS/FAIL is genuinely sensitive to the direction of the update,
not to numerical noise.

## Contents

| File | Purpose |
| --- | --- |
| `learning_direction.py` | enumeration helpers, the deterministic training loop and the CLI |
| `README.md` / `README_ZH.md` | this document and its Chinese translation |

## Dependencies

Only the project requirements (`torch`, `numpy`).
