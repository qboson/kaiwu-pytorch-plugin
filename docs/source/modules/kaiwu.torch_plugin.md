# 模块手册

## Module contents

```{eval-rst}
.. automodule:: kaiwu.torch_plugin
   :members:
   :undoc-members:
   :show-inheritance:
```

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

## MAIFS training checkpoints

`FeatureSelectionWrapper.state_dict()` includes the number of completed
`fit_weights` epochs. Reconstruct the wrapper with the same
`mask_update_epochs` and load its state to continue the periodic feature-mask
schedule. For example, a checkpoint after epoch two with an interval of three
updates the mask after the first resumed epoch, rather than waiting three more.
Optimizer state and the wrapper configuration must still be saved separately.
Legacy checkpoints without the epoch count load normally and start a fresh
schedule at zero.
