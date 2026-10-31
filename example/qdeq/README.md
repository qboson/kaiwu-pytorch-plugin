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

## Background

`QDEQ` targets models that must run under tight storage and low-bit
constraints, for example forecasting on edge devices:

- **Fewer parameters.** A DEQ obtains depth by repeating one shared layer
  until a fixed point is reached, instead of stacking independent layers, so
  only one layer's weights are stored. Training memory does not grow with the
  number of iterations either.
- **Discrete update decisions.** The usual DEQ solver, Anderson acceleration,
  mixes past iterates with coefficients computed in the continuous real
  domain. `QDEQ` restricts each update coefficient to a `coeff_bits`-bit grid
  and selects the combination by solving a QUBO. The update decision itself
  becomes low-bit, and it can be executed by a combinatorial optimization
  backend such as Kaiwu SA or the Kaiwu CIM physical solver.
- **Works with quantized layers.** The fixed-point solve and the final
  gradient-carrying layer call are separate stages, so the shared layer can be
  quantized in either stage. See [Use with quantized models](#use-with-quantized-models).

## Reproducible QDEQ random streams

Pass a caller-owned generator to isolate QDEQ's random search directions and
Jacobian-regularizer probes from unrelated Torch draws:

```python
rng = torch.Generator(device="cpu").manual_seed(17)
model = QDEQ(layer, sampler, config, generator=rng)
```

The generator advances on each draw; reusing the same object continues its
stream instead of repeating one seed on every call. For checkpoint continuation,
save both `model.state_dict()` and `rng.get_state()`, then restore them with
`model.load_state_dict(...)` and `rng.set_state(...)`. Generator state is not
automatically included in the model state dictionary. A failed solve can consume
draws too; restore a saved generator state if a retry must replay them.

The generator must match the tensors' device. Moving the model does not move
the generator. With `generator=None` (the default), existing global-RNG behavior
is unchanged. This option does not seed the Kaiwu optimizer, random operations
inside the supplied layer, or a subclass's custom forward solver. Those need
their own reproducibility controls; cross-device bitwise equivalence is not
promised. Direct calls to `QDEQ.jacobian_loss` also accept `generator=`.

Run the CPU checkpoint/replay example after installing the plugin:

```bash
python example/qdeq/simple/reproducible_cpu.py
```

It verifies replayed outputs and regularizer probes, an unchanged global Torch
stream, and finite training gradients. It uses exhaustive enumeration of a tiny
Ising problem, with no dataset, quantum hardware, or cloud task.

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

`simple_train_example.py` with default arguments (2 epochs) finishes in a few
minutes on a CPU and reaches a test MSE of about 0.43 in the standardised space.

## Results: Kaiwu CIM vs Anderson

To check the Ising solver on physical hardware, the same trained QDEQ
checkpoint was evaluated with two forward solvers: the Kaiwu CIM backend and
the classical Anderson solver. Setting:

- model: iTransformer (`d_model = 256`) with one shared `QDEQ` encoder layer, FP32
- data: ETTh1, 96-step look-back, 96-step horizon
- samples: the first 128 test windows in order (4 batches of 32)
- Ising step: 8 search directions × 10 bits per coefficient

| Forward solver | MSE | MAE |
| --- | ---: | ---: |
| Anderson | 0.28730 | 0.35111 |
| Kaiwu CIM | 0.28728 | 0.35110 |

The two solvers give the same forecasts to within 0.01% in MSE, so the fixed
points found through CIM match the classical solver. The iTransformer example
used here will be added in a follow-up.

## Use with quantized models

`QDEQ` provides two hooks that a subclass can override to switch the state of
the shared layer:

- `solve_context()`: entered around the whole fixed-point solve, which runs
  without gradients
- `reattach_context()`: entered around the final layer call at the fixed point,
  which carries the gradient

For example, to run the shared layer with W8A8 fake quantization (8-bit
weights and activations) when re-attaching the gradient:

```python
class QuantizedQDEQ(QDEQ):
    def reattach_context(self):
        return self.layer.quantized()   # a context manager that enables fake quantization
```

Weights then only need to be stored in 8 bits. In our iTransformer experiments
on five forecasting datasets, combining the shared DEQ layer with W8A8 reduced
static weight storage by 4.3×–12.8× compared with the explicit multi-layer
model. An iTransformer example with W8A8 quantization will be added in a
follow-up.

## How the Ising solver works

Each forward step:

1. computes the residual `r = layer(z, x) - z`
2. takes `num_basis` search directions `V` (`-r`, past residuals and past updates)
3. estimates `G = J V` by finite differences
4. encodes the coefficients `a` of `min_a ||r + G a||^2` with `coeff_bits` bits
   each, converts the QUBO to an Ising matrix with one auxiliary spin through
   `kaiwu.torch_plugin._kaiwu_compat.qubo_matrix_to_ising_matrix`, and calls `sampler.solve`
5. decodes the best solution and updates `z ← z + V a`
