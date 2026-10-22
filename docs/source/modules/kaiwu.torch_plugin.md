# 模块手册

## Module contents

```{eval-rst}
.. automodule:: kaiwu.torch_plugin
   :members:
   :undoc-members:
   :show-inheritance:
```


## Full BM Upper-Triangle Energy Evaluation

For the default symmetry definition, float32/float64 `BoltzmannMachine.forward`
uses the strict upper triangle U directly:
`-s @ bias - sum((s @ U) * s)` is mathematically the same energy as
`-s @ bias - 0.5 * sum((s @ (U + U.T)) * s)`.
This avoids constructing the extra symmetric N×N tensor on every energy call.
The public symmetry method remains available for Ising conversion and sampling.

Subclass, instance and class overrides of `symmetrized_quadratic_coef` retain
the original energy path. Autocast and float16/bfloat16 parameters or states
also retain that path. Ordinary module hooks and gradient contexts continue to
apply. The optimization changes neither parameter storage nor sampling rules.

Floating-point operations are regrouped: ordinary energy/input/parameter
comparisons use dtype-appropriate tolerances rather than bitwise equality.
Near ties can change downstream decisions. Extreme coefficients can have
different overflow or cancellation behavior; a single finite pair no longer
needs to be doubled before halving. This is not a promise of exact arithmetic
or universal numerical improvement.

CPU allocation profiling verifies that one N×N float32 symmetric-add allocation
is removed (16 MiB at 2048 nodes). The triangular tensor and dense model
parameters remain. These are operator allocation counts, not process peak RSS
measurements. `benchmarks/bm_upper_energy.py` reports no-grad forward and
forward/backward timings separately; they do not represent end-to-end training,
external sampling or QPU speedups.
