# 模块手册

## Module contents

```{eval-rst}
.. automodule:: kaiwu.torch_plugin
   :members:
   :undoc-members:
   :show-inheritance:
```

## MAIFS feature-count settings

`FeatureSelectionWrapper` starts with every feature selected. The wrapper applies the feature-count bounds when it updates the mask. An update can be triggered with `update_mask()` or scheduled during `fit_weights()`.

`cardinality_k` is a soft target in the QUBO objective, so an updated mask may contain a different number of features. To require exactly `k` features after each mask update, set both `min_selected_features` and `max_selected_features` to `k`.

An explicit minimum applies to every updated mask. If the minimum is omitted, the wrapper uses its default minimum only when the mask is empty after applying the upper bound.

## MAIFS QUBO and SDK energy conventions

`QuadraticLinearSolver.solve(Q, l)` represents the binary objective
`0.5 * x.T @ Q @ x + l @ x`. Its public
`qubo_matrix_to_ising_matrix(U)` helper takes an upper-triangular QUBO coefficient
matrix and returns an upper-triangular spin coefficient matrix `M`. The positive
spin objective `s.T @ M @ s` equals the QUBO objective up to an additive constant,
where `x = (s[:-1] * s[-1] + 1) / 2`. Both global spin gauges encode the same
binary state. The local-search backend minimizes this positive objective.

Kaiwu SDK 1.3.1 evaluates its Hamiltonian as `-s.T @ J @ s`. MAIFS therefore
submits `J = -(M + M.T) / 2` to SA and CIM. This preserves the QUBO energy
differences and supplies the symmetric matrix required by SA's single-spin
energy updates. CIM receives this conversion before precision adaptation and
variable splitting, so splitting penalties retain their intended sign.
Precision settings still control quantization; the conversion does not make a
heuristic solver exact.
