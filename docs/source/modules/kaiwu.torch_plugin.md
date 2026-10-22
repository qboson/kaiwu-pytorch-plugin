# 模块手册

## Module contents

```{eval-rst}
.. automodule:: kaiwu.torch_plugin
   :members:
   :undoc-members:
   :show-inheritance:
```

## MAIFS Binary States and Solver Results

`maifs.qubo.solve_qubo` requires an initial vector containing exact numeric
`0` or `1` values, with one entry for each QUBO variable. Values are checked
before integer normalization. Boolean arrays, exact floating-point values and
numeric object arrays remain accepted. Fractional values such as `0.9`, `1.9`
and `-0.2`, numeric strings, NaN and infinity raise `ValueError`; they are not
rounded or truncated into a different initial state.

The selected backend solution must contain exact numeric `-1` or `1` spins,
including one auxiliary spin. The first solution is decoded using the
auxiliary spin's gauge: a variable is active when its spin equals that
auxiliary spin. Both gauges and exact floating-point spins are supported.
Malformed SA results such as `[1.9, -1.9, 1.9]` raise `RuntimeError` before
integer normalization. There is no tolerance that turns a nearby value into
a valid spin.

`FeatureSelectionWrapper.update_mask` validates the backend's selected result
before projecting feature-count bounds or storing a new mask. When a backend
returns invalid spins, the existing feature mask remains unchanged.
