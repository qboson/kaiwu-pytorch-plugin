# Language: [中文](README_ZH.md) | [English](README.md)

# `example/dbm`

This directory contains a self-contained example for `DeepBoltzmannMachine`
(DBM), demonstrating multi-modal binary pattern learning, denoising
reconstruction, and one-shot joint multi-layer Ising sampling.

## Model Overview

Unlike a greedily stacked Deep Belief Network (`UnsupervisedDBN`),
`DeepBoltzmannMachine` defines a single undirected multi-layer Hamiltonian over
the visible layer `v` and hidden layers `h^(1), ..., h^(L)`:

$$E(v, h^{(1)}, \dots, h^{(L)}) = - b_v^T v - \sum_{\ell=1}^{L} b_{h^{(\ell)}}^T h^{(\ell)} - v^T W^{(1)} h^{(1)} - \sum_{\ell=2}^{L} (h^{(\ell-1)})^T W^{(\ell)} h^{(\ell)}$$

- **Positive phase**: variational mean-field inference (`mean_field_inference`)
  with monotonically non-increasing variational free energy.
- **Negative phase**: `to_ising_matrix()` / `get_ising_matrix()` maps **all
  layers jointly** into a single `(num_nodes + 1, num_nodes + 1)` Ising matrix
  so Kaiwu SA or CIM samples `(v, h^(1), ..., h^(L))` together in one call.

## Quick Start

No external datasets or additional dependencies are required. From the
repository root:

```bash
pip install -e .
python example/dbm/run_dbm.py
```

Optional flags:

- `--epochs`: number of training epochs (default `80`)
- `--lr`: Adam learning rate (default `0.1`)
- `--seed`: random seed (default `42`)
- `--sampler`: Ising sampler backend (`local` license-free NumPy SA by default,
  `sa` for Kaiwu `SimulatedAnnealingOptimizer`, or `cim` for `CIMOptimizer` +
  `PrecisionReducer`; `sa`/`cim` read `USER_ID` and `SDK_CODE` from the
  environment when a Kaiwu license is required)

## What It Demonstrates

`run_dbm.py` trains a 2-hidden-layer DBM (`num_visible=12`,
`hidden_dims=(8, 4)`) on 4 structured 12-bit modes (pairwise Hamming distance
6). On CPU it completes in a few seconds and reports before/after metrics:

- **Variational free-energy gap** `F_MF(v_ood) - F_MF(v_in)`
- **Denoising reconstruction accuracy** (bit-level & exact) from 2-bit-flipped
  inputs via multi-layer mean-field inference
- **Joint Ising generation validity rate** (fraction of `dbm.sample(sampler)`
  visible states matching learned prototypes)

Sample CPU output (`--seed 42`, `--sampler local`, ~4.1 s):

| Stage | `vfe_gap` | `denoise_bit_acc` | `denoise_exact_acc` | `gen_valid_rate` |
| --- | ---: | ---: | ---: | ---: |
| Init | `0.000` | `47.92%` | `0.00%` | `0.00%` |
| Final (80 epochs) | `13.200` | `100.00%` | `100.00%` | `100.00%` |
