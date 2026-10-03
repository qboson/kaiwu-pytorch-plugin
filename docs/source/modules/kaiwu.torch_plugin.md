# 模块手册

## Module contents

```{eval-rst}
.. automodule:: kaiwu.torch_plugin
   :members:
   :undoc-members:
   :show-inheritance:
```

## MAIFS Ordered QUBO-to-Ising Conversion

`QuadraticLinearSolver.qubo_matrix_to_ising_matrix` uses two ordered NumPy
cumulative sums instead of Python pair loops. For each auxiliary-spin field,
it accumulates incoming upper-triangle pairs, adds half the QUBO diagonal,
then accumulates outgoing pairs, in the same order as the scalar implementation.
It does not regroup these contributions into independent row and column sums,
which can change cancellation and near-tie behavior.

The output remains float64, upper triangular, and includes one auxiliary spin.
Finite square inputs, including noncontiguous and read-only NumPy arrays, are
accepted without modifying the input. Lower-triangle entries are ignored in the
conversion but still subject to finite-value validation. Finite extreme inputs
can overflow the output as before; caller overflow/invalid error policies are
honored, and scalar multiplication underflow remains silent.

This is a CPU conversion optimization, with O(N²) work and storage. It adds one
reused float64 N×N scratch array (2 MiB at N=512, 8 MiB at N=1024); constructing
the upper triangle also uses a temporary boolean mask and linear-size arithmetic
temporaries. These element-size estimates are not a measurement or bound on
process peak RSS. It is not a memory optimization or an end-to-end solver
speedup claim. `benchmarks/maifs_ising_conversion.py` compares full conversions
with the original scalar code and verifies output equality before timing.
