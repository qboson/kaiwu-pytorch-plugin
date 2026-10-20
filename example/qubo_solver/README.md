# QUBO Solver: Max-Cut with `QuadraticLinearSolver` and `solve_qubo`

This example demonstrates the public MAIFS QUBO helpers on a small Max-Cut
problem. It is self-contained and runs on CPU without a Kaiwu license because
it uses the built-in `local_search` backend.

## What it shows

1. Build the QUBO

   \[
   \min_{x \in \{0,1\}^n} \frac{1}{2} x^\top Q x + b^\top x
   \]

   for the Max-Cut objective on a cycle graph. For one edge `(i, j)` the cut
   indicator is `x_i + x_j - 2 x_i x_j`, so maximizing the cut is equivalent to
   minimizing `2 x_i x_j - x_i - x_j`.

2. Convert the QUBO to the auxiliary-spin Ising matrix with
   `QuadraticLinearSolver().solve(Q, b)`.

3. Solve it with `solve_qubo(..., solver="local_search")`.

4. Verify the result by exhaustive enumeration, which is feasible for the
   small example graphs used here.

## Run

From the repository root:

```bash
python example/qubo_solver/run_qubo_solver.py --nodes 5
```

Example output (values depend on the graph):

```text
Max-Cut QUBO on a cycle graph
nodes=5
edges=((0, 1), (1, 2), (2, 3), (3, 4), (4, 0))
Ising matrix shape: (6, 6)
initial state: [0 0 0 0 0] value=0.000000
local-search state: [1 0 1 0 0] value=-4.000000
brute-force state: [0 0 1 0 1] value=-4.000000
```

The local-search state can differ from the brute-force state when several
optimal cuts exist; both have the same QUBO value.

## Backends

`solve_qubo` also accepts `solver="sa"` and `solver="kaiwu_cim"`. These
backends use the Kaiwu SDK and require the corresponding license and setup.
The local-search backend is always available and is the default used here.

## Files

- `run_qubo_solver.py`: runnable example and command-line entry point.
- `../../tests/test_qubo_solver_example.py`: verifies the example against
  exhaustive enumeration and checks the QUBO formulation.

For the full feature-selection workflow built on the same solver helpers, see
`example/feature_selection/`.
