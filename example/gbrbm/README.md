**Language Versions**: [中文](README_ZH.md) | [English](README.md)

### Training a Gaussian-Bernoulli RBM on Continuous Data

This example shows the complete workflow of `GaussianBernoulliRestrictedBoltzmannMachine` on real-valued data — the model to use when your inputs are continuous (sensor readings, image intensities, physical measurements) instead of binary.

Run it with:

```bash
python example/gbrbm/run_gbrbm.py
```

The script:

* builds a synthetic correlated continuous dataset;
* trains the model with contrastive divergence: the positive phase completes the data with sampled Bernoulli units (`infer_from_gaussian`), the negative phase draws one model sweep with `gibbs_sample`, and `bm.objective(s_positive, s_negative)` — whose gradient equals the gradient of the negative log-likelihood — is minimized with plain SGD;
* generates fresh samples from the trained model with a burned-in Gibbs chain (`gibbs_sample` with `n_burnin`);
* compares per-dimension means and standard deviations of data versus generated samples, and scores both with `marginal_energy` (the free energy of Gaussian states).

Expected output (seeded, CPU):

```text
training on continuous data (local Gibbs negative phase)...
epoch  100  objective     0.1418
epoch  200  objective    -0.0345
epoch  300  objective     0.0182
objective: start 1.9143 -> end 0.0182

per-dimension mean (data vs generated):
  dim 0:   0.9981    1.0140
  dim 1:  -0.3382   -0.2790
  dim 2:   0.3734    0.3448
  dim 3:  -1.2853   -1.3006

per-dimension std (data vs generated):
  dim 0:   1.0176    1.0236
  dim 1:   1.1600    1.1729
  dim 2:   1.1005    1.1170
  dim 3:   1.0358    1.0590

marginal energy (free energy of Gaussian states): data -0.8093, generated -0.7373
```

The Gaussian energy of the model is diagonal in the visible units, so the match to target is the per-dimension mean and variance, not the off-diagonal correlations of the data.

**Dependencies**: none beyond the package requirements (`torch`, `kaiwu`). The default path runs the negative phase with local Gibbs sampling, so **no Kaiwu SDK license is needed**.

#### Using the Kaiwu SDK sampler

The quantum-plugin path replaces the local Gibbs sweep of the negative phase with `bm.sample(sampler)`, which converts the Bernoulli side of the model to an Ising problem and solves it with the Kaiwu SDK. Set `USE_SOLVER = True` at the top of `run_gbrbm.py`, install the SDK (`pip install kaiwu==1.3.1`), and initialize your license as in `example/run_rbm.py`:

```python
import os
import kaiwu as kw

kw.license.init(os.getenv("USER_ID"), os.getenv("SDK_CODE"))
```

Credentials are obtained from [platform.qboson.com](https://platform.qboson.com) (see the [installation guide](https://kaiwu-pytorch-plugin.readthedocs.io/en/latest/source/getting_started/installation.html) for details).
