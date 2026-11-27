# Language: [中文](README_ZH.md) | [English](README.md)

# `example/crbm`

This directory contains a self-contained example for
`ConditionalRestrictedBoltzmannMachine` (CRBM), demonstrating conditional
binary pattern generation and completion with a Kaiwu Ising sampler.

## Model Overview

`ConditionalRestrictedBoltzmannMachine` extends `AbstractBoltzmannMachine` with
external conditioning inputs `c` (binary or continuous):

$$E(v, h \mid c) = - v^T W h - c^T U h - b_v^T v - b_h^T h - c^T A v$$

- `A` (`cond_visible_coef`) can be toggled with `use_cond_visible=True/False`.
- Folding `c` into effective linear biases `b_v_eff = b_v + c @ A` and
  `b_h_eff = b_h + c @ U` produces an equivalent Ising matrix via
  `condition_ising_matrix(c)`, enabling direct conditional sampling with
  `SimulatedAnnealingOptimizer` or `CIMOptimizer`.

## Quick Start

No external datasets or additional dependencies are required. From the
repository root:

```bash
pip install -e .
python example/crbm/run_crbm.py
```

Optional flags:

- `--epochs`: number of training epochs (default `35`)
- `--lr`: SGD learning rate (default `0.2`)
- `--seed`: random seed (default `42`)
- `--sampler`: Ising sampler backend (`local` license-free NumPy SA by default,
  `sa` for Kaiwu `SimulatedAnnealingOptimizer`, or `cim` for `CIMOptimizer` +
  `PrecisionReducer`; `sa`/`cim` read `USER_ID` and `SDK_CODE` from the
  environment when a Kaiwu license is required)

## What It Demonstrates

`run_crbm.py` trains a CRBM (`num_visible=8`, `num_condition=4`,
`num_hidden=6`) on 4 structured 8-bit patterns conditioned on one-hot class
labels. On CPU it completes in a few seconds and reports:

- **Free-energy gap** `F(v_wrong | c) - F(v_true | c)` before vs. after training
- **Conditional generation exact & bit accuracy** from `condition_sample`
- **Conditional pattern completion accuracy** from 25% bit-flipped inputs
