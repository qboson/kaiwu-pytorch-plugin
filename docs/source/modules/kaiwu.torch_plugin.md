# 模块手册

## Module contents

```{eval-rst}
.. automodule:: kaiwu.torch_plugin
   :members:
   :undoc-members:
   :show-inheritance:
```

## MAIFS Zero-Coefficient Precision Plans

A feature-selection QUBO can have only zero coefficients, for example when a
zero-weight linear model has zero mask gradients and Hessian. Every binary mask
then has the same QUBO energy. The CIM precision path preserves this valid flat
objective as an integer zero matrix before applying the SDK's split and restore
helpers.

For an all-zero matrix, `PrecisionSplitPlan.precision_info` reports
`{"precision": 1, "multiplier": 1.0}`: one bit can encode zero, and unit scaling
leaves every coefficient unchanged. The adjusted matrix is a separate array from
the input. This avoids the pinned Kaiwu 1.3.1 precision helpers' degenerate return
for zero coefficients; nonzero matrices retain their SDK precision behavior.
