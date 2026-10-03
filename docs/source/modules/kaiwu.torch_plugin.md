# 模块手册

## Module contents

```{eval-rst}
.. automodule:: kaiwu.torch_plugin
   :members:
   :undoc-members:
   :show-inheritance:
```


## MAIFS Precision Search and Failed Searches

`PrecisionSplitExplorer.search` selects the highest feasible **source** precision
in the configured inclusive interval. The target precision of the split matrix
is unchanged. Kaiwu preprocessing can produce nonmonotone split sizes: an
infeasible precision does not rule out a higher feasible one.

The search probes the configured coarse sequence and the upper endpoint, then
checks untested precisions above the best coarse result in descending order.
Each precision is attempted at most once, up to
`max_precision - start_precision + 1` preprocessing calls. `precision_step`
controls coarse spacing, rather than an early-stop rule. Search history records
actual `coarse` and `fine` attempts; it need not be sorted by precision.

A new search invalidates the previous plan before validating its input. If
validation, preprocessing or feasibility search fails, `restore_solution` raises
until a later search succeeds. History contains only completed attempts from the
latest call. A previously returned plan remains a separate result object.

More complete search can require additional preprocessing time. The explorer
retains at most the best plan and the current candidate, without caching all
split matrices. `max_bits` restricts accepted plans; it is **not** a bound on
SDK temporary allocations while constructing a candidate. Use a practical
precision interval for available memory; no licensed sampler or QPU is required
to run precision preprocessing tests.
