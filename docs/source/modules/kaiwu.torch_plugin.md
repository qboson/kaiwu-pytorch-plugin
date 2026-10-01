# 模块手册

## Module contents

```{eval-rst}
.. automodule:: kaiwu.torch_plugin
   :members:
   :undoc-members:
   :show-inheritance:
```

## MAIFS simulated annealing seeds

For `solve_qubo(..., solver="sa")` and `FeatureSelectionWrapper(solver="sa")`,
`random_state` defaults to `0` and is passed to the SDK optimizer as `rand_seed`.
For example, use `solver_kwargs={"random_state": 17}` on the wrapper to seed its
SA calls. MAIFS does not reseed NumPy's global RNG. With SDK 1.3.1 and the default
`process_num=1`, an integer seed selects a local optimizer RNG, preserving the
caller’s NumPy random stream on both successful and failed solver calls.

An explicit SDK `rand_seed` takes precedence over `random_state`; other SDK
optimizer options also keep their supplied values. This includes
`rand_seed=None`, which asks SDK 1.3.1 to choose a seed from NumPy's global RNG.
SDK multi-process execution may also reseed that global RNG. These explicitly
requested SDK behaviors are preserved, so the random-stream guarantee applies
to single-process calls with an integer seed.
