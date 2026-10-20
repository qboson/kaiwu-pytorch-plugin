# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Solve a small Max-Cut QUBO with the public MAIFS helpers.

The example is intentionally self-contained: it uses the built-in
``local_search`` backend, so no Kaiwu license or quantum hardware is needed.
It shows the same QUBO -> Ising conversion that ``FeatureSelectionWrapper``
uses internally.
"""

from __future__ import annotations

import argparse
from itertools import product

import numpy as np

from kaiwu.torch_plugin import QuadraticLinearSolver
from kaiwu.torch_plugin.maifs.qubo import solve_qubo


def cycle_edges(num_nodes: int) -> tuple[tuple[int, int], ...]:
    """Return the undirected edges of a cycle on ``num_nodes`` nodes."""
    if num_nodes < 2:
        return ()
    return tuple((node, (node + 1) % num_nodes) for node in range(num_nodes))


def build_max_cut_qubo(
    num_nodes: int,
    edges: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Build the QUBO ``minimize 0.5 x^T Q x + b^T x`` for Max-Cut.

    For binary ``x`` the cut indicator of one edge ``(i, j)`` is
    ``x_i + x_j - 2 x_i x_j``.  Maximizing the cut is therefore equivalent to
    minimizing ``2 x_i x_j - x_i - x_j``.

    Args:
        num_nodes: Number of graph nodes.
        edges: Undirected edges as ``(left, right)`` pairs.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Quadratic matrix and linear term.

    Raises:
        ValueError: If the node count is invalid or an edge is malformed.
    """
    if num_nodes < 1:
        raise ValueError("num_nodes must be at least 1")

    quadratic = np.zeros((num_nodes, num_nodes), dtype=float)
    linear = np.zeros(num_nodes, dtype=float)
    for left, right in edges:
        if left == right:
            raise ValueError("self-loops are not valid Max-Cut edges")
        if not (0 <= left < num_nodes and 0 <= right < num_nodes):
            raise ValueError("edge endpoint is outside the node range")
        quadratic[left, right] += 2.0
        quadratic[right, left] += 2.0
        linear[left] -= 1.0
        linear[right] -= 1.0
    return quadratic, linear


def qubo_objective(
    quadratic: np.ndarray,
    linear: np.ndarray,
    state: np.ndarray,
) -> float:
    """Evaluate the QUBO objective for one binary state."""
    state = np.asarray(state, dtype=float)
    return float(0.5 * state @ quadratic @ state + linear @ state)


def brute_force_solution(
    quadratic: np.ndarray,
    linear: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Enumerate every binary state and return the global optimum."""
    num_variables = linear.shape[0]
    best_state = np.zeros(num_variables, dtype=int)
    best_value = float("inf")
    for bits in product((0, 1), repeat=num_variables):
        state = np.asarray(bits, dtype=int)
        value = qubo_objective(quadratic, linear, state)
        if value < best_value:
            best_state = state
            best_value = value
    return best_state, best_value


def solve_demo(num_nodes: int = 5, max_iter: int = 200) -> dict:
    """Solve one cycle Max-Cut QUBO and compare it with exhaustive search.

    Args:
        num_nodes: Number of graph nodes in the cycle.
        max_iter: Greedy-improvement budget for the local-search backend.

    Returns:
        dict: Problem data, Ising matrix, and initial/optimized/reference values.
    """
    edges = cycle_edges(num_nodes)
    quadratic, linear = build_max_cut_qubo(num_nodes, edges)
    initial_state = np.zeros(num_nodes, dtype=int)
    initial_value = qubo_objective(quadratic, linear, initial_state)

    ising_matrix = QuadraticLinearSolver().solve(quadratic, linear)
    best_state = solve_qubo(
        quadratic,
        linear,
        initial_state,
        solver="local_search",
        max_iter=max_iter,
    )
    best_value = qubo_objective(quadratic, linear, best_state)
    brute_force_state, brute_force_value = brute_force_solution(quadratic, linear)

    return {
        "num_nodes": num_nodes,
        "edges": edges,
        "quadratic": quadratic,
        "linear": linear,
        "ising_matrix": ising_matrix,
        "initial_state": initial_state,
        "initial_value": initial_value,
        "best_state": best_state,
        "best_value": best_value,
        "brute_force_state": brute_force_state,
        "brute_force_value": brute_force_value,
    }


def main() -> None:
    """Run the example from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nodes",
        type=int,
        default=5,
        help="number of nodes in the cycle graph",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="greedy-improvement budget for local search",
    )
    args = parser.parse_args()

    result = solve_demo(num_nodes=args.nodes, max_iter=args.max_iter)
    print("Max-Cut QUBO on a cycle graph")
    print(f"nodes={result['num_nodes']}")
    print(f"edges={result['edges']}")
    print("QUBO matrix:")
    print(result["quadratic"])
    print(f"linear term: {result['linear']}")
    print(f"Ising matrix shape: {result['ising_matrix'].shape}")
    print(
        "initial state: "
        f"{result['initial_state']} value={result['initial_value']:.6f}"
    )
    print(
        "local-search state: "
        f"{result['best_state']} value={result['best_value']:.6f}"
    )
    print(
        "brute-force state: "
        f"{result['brute_force_state']} value={result['brute_force_value']:.6f}"
    )
    if not np.isclose(result["best_value"], result["brute_force_value"]):
        print(
            "note: local search stopped at a local optimum; "
            "increase --max-iter or use another backend."
        )


if __name__ == "__main__":
    main()
