# Language: [中文](README_ZH.md) | [English](README.md)

# `example/qdeq`

This directory contains examples for `QDEQ`, the quantum-assisted deep
equilibrium layer. `QDEQ` turns a weight-tied layer `layer(z, x)` into a layer
that returns its fixed point `z* = layer(z*, x)`: the fixed point is solved
without building an autograd graph and gradients come from the implicit
function theorem, so memory does not grow with the number of solver steps.

The coefficients of every forward fixed-point update are encoded as a small
QUBO and solved by a Kaiwu SDK optimizer (simulated annealing or CIM). The
optimizer is created by the caller and passed in, in the same way as the
`sampler` of `QVAE` and `EnergyModel`. The backward pass solves the
implicit-gradient equation with Anderson acceleration.

## Data

The example uses the ETTh1 electricity transformer dataset
([ETDataset](https://github.com/zhouhaoyi/ETDataset)). Download
`ETT-small/ETTh1.csv` and place it at:

```text
example/qdeq/data/ETTh1.csv
```

Use `--data-path` for another location.

## Quick start

```bash
pip install -e .        # install the plugin from the repository root
pip install -r example/qdeq/requirements.txt
python example/qdeq/simple/simple_train_example.py --sampler sa
python example/qdeq/simple/simple_forecast_example.py --sampler sa
```

`--sampler` accepts:

- `sa`: Kaiwu simulated annealing (`kaiwu.classical.SimulatedAnnealingOptimizer`), runs locally
- `cim`: Kaiwu CIM (`CIMOptimizer` + `PrecisionReducer`), requires a licensed Kaiwu account
- `anderson`: classical baseline, `AndersonDEQ` in `etth1_common.py`, which
  subclasses `QDEQ` and overrides `solve_fixed_point` with Anderson acceleration

## Simple

`simple/` is a minimal runnable example on ETTh1 (96-step look-back, 96-step horizon):

- `simple/etth1_common.py`: data split, model, sampler construction and the Anderson baseline
- `simple/simple_train_example.py`: trains and saves a checkpoint
- `simple/simple_forecast_example.py`: loads the checkpoint and evaluates the test split with the chosen solver

The model follows iTransformer: the history of each variable is embedded as one
token and the encoder is one `QDEQ` layer around a weight-tied
variable-attention block. To finish within minutes, training windows are taken
with stride 4 and each epoch trains on 60 batches.

```python
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import QDEQ, QDEQConfig

sampler = SimulatedAnnealingOptimizer()
deq = QDEQ(block, sampler, QDEQConfig(f_thres=12, num_basis=4, coeff_bits=4))
z = deq(tokens)            # called like any other layer
deq.get_last_stats()       # forward/backward steps, residuals, sampler time
```

## Example results

`simple_train_example.py` with default arguments (2 epochs); errors are
measured in the standardised space:

| Forward solver | test MSE | test MAE | mean forward steps |
| --- | ---: | ---: | ---: |
| Anderson | 0.4263 | 0.4312 | 6.3 |
| Kaiwu SA | 0.4280 | 0.4326 | 9.2 |

Weights trained with SA reach the same test MSE (0.4280) whether evaluated with
SA or Anderson, so the Ising solver converges to the same fixed point.

## How the Ising solver works

Each forward step:

1. computes the residual `r = layer(z, x) - z`
2. takes `num_basis` search directions `V` (`-r`, past residuals and past updates)
3. estimates `G = J V` by finite differences
4. encodes the coefficients `a` of `min_a ||r + G a||^2` with `coeff_bits` bits
   each, converts the QUBO to an Ising matrix with one auxiliary spin through
   `kaiwu.conversion.qubo_matrix_to_ising_matrix`, and calls `sampler.solve`
5. decodes the best solution and updates `z ← z + V a`
