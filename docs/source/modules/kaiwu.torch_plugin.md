# 模块手册

## Module contents

```{eval-rst}
.. automodule:: kaiwu.torch_plugin
   :members:
   :undoc-members:
   :show-inheritance:
```

## Conditional BM sampling

`BoltzmannMachine.condition_sample` prepares the static hidden coupling matrix,
column sums and hidden-visible couplings once per call, then constructs an
independent conditioned matrix for each visible row. Both binary and soft visible
conditions keep their existing semantics. Couplings are recomputed on each new
call, including after parameter updates; no persistent cache is retained.

Solvers may modify their input matrices without affecting later rows. The solver
is still called once per condition, and variable numbers of returned solutions
are concatenated in input order. This optimization reduces matrix-construction
work; solver and network time are separate costs.
