# 模块手册

## 采样与数值精度

BM、RBM 和 GBRBM 的 `sample(sampler)` 使用模型参数的数据类型构造返回的
PyTorch 状态。传给采样器的 Ising 矩阵是 NumPy 数组；float16、float32 和
float64 矩阵保留各自精度。NumPy 不支持 BF16，因此 BF16 模型仅在导出
Ising 矩阵时转为 float32，模型参数和返回状态仍为 BF16。

FullBM 的 `condition_sample` 保留已有的 `dtype=torch.float32` 参数默认值。
使用 BF16 等其他精度评分条件样本时，应显式传入 `dtype=model.linear_bias.dtype`。
该接口精度转换不改变采样器的分布、求解算法或硬件授权要求。

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
