# QUBO 求解示例：用 `QuadraticLinearSolver` 与 `solve_qubo` 求解 Max-Cut

本示例演示 MAIFS 的公开 QUBO 辅助接口。示例完全自包含，使用内置的
`local_search` 后端，在 CPU 上即可运行，不需要 Kaiwu 许可证。

## 示例内容

1. 为环图上的 Max-Cut 构造 QUBO：

   \[
   \min_{x \in \{0,1\}^n} \frac{1}{2} x^\top Q x + b^\top x
   \]

   对一条边 `(i, j)`，割指示量为 `x_i + x_j - 2 x_i x_j`，因此最大化割与
   最小化 `2 x_i x_j - x_i - x_j` 等价。

2. 使用 `QuadraticLinearSolver().solve(Q, b)` 把 QUBO 转换为带辅助自旋的
   Ising 矩阵。

3. 使用 `solve_qubo(..., solver="local_search")` 求解。

4. 用穷举法验证结果（示例图的变量数很小，可以完整枚举）。

## 运行

在仓库根目录执行：

```bash
python example/qubo_solver/run_qubo_solver.py --nodes 5
```

输出示例（具体数值取决于图）：

```text
Max-Cut QUBO on a cycle graph
nodes=5
edges=((0, 1), (1, 2), (2, 3), (3, 4), (4, 0))
Ising matrix shape: (6, 6)
initial state: [0 0 0 0 0] value=0.000000
local-search state: [1 0 1 0 0] value=-4.000000
brute-force state: [0 0 1 0 1] value=-4.000000
```

如果存在多个最优割，局部搜索得到的解可能与穷举解不同，但 QUBO 目标值相同。

## 后端

`solve_qubo` 还支持 `solver="sa"` 和 `solver="kaiwu_cim"`。这两个后端依赖
Kaiwu SDK，需要相应的许可证与环境配置。本示例默认使用的 `local_search`
后端始终可用。

## 文件

- `run_qubo_solver.py`：可运行示例与命令行入口。
- `../../tests/test_qubo_solver_example.py`：用穷举结果验证示例，并检查 QUBO
  构造是否与割目标一致。

完整的特征选择工作流（同样基于这些求解器接口）见
`example/feature_selection/`。
